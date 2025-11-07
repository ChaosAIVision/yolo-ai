"""Simplified YOLO API v1 - YouTube streaming with YOLO detection via WebSocket."""
import asyncio
import base64
import io
import json
import logging
import os
import subprocess
import time

import aiohttp
import cv2
import numpy as np
from aiohttp import web, WSMsgType
from PIL import Image

from src.config import (
    BENTO_ENDPOINT_URL,
    FPS_LIMIT,
    FRAME_SKIP,
    CLASS_NAMES,
)

logger = logging.getLogger(__name__)


class YouTubeStreamer:
    """YouTube streamer with YOLO detection."""

    def __init__(self, youtube_url: str):
        self.youtube_url = youtube_url
        self.frame_counter = 0
        self.last_annotated_frame = None
        self.ffmpeg_process = None
        self.width = 640
        self.height = 480

        # Use yt-dlp to get stream URL and ffmpeg to decode
        try:
            import yt_dlp
            
            logger.info(f"Extracting YouTube info for: {youtube_url}")
            ydl_opts = {
                'format': 'best[height<=720]/bestvideo[height<=720]+bestaudio/best[height<=720]',
                'quiet': True,
                'no_warnings': True,
                'no_check_certificate': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
            logger.info(f"YouTube info extracted successfully")
            
            # Try to get stream URL from formats
            stream_url = None
            formats = info.get('formats', [])
            
            def sort_key(fmt):
                vcodec = fmt.get('vcodec', 'none')
                height = fmt.get('height') or 0
                return (vcodec == 'none', -height)
            
            for fmt in sorted(formats, key=sort_key):
                url = fmt.get('url', '')
                vcodec = fmt.get('vcodec', 'none')
                protocol = fmt.get('protocol', '')
                if vcodec != 'none' and protocol in ['http', 'https'] and url:
                    if 'manifest' not in url.lower() and 'm3u8' not in url.lower() and 'dash' not in url.lower():
                        stream_url = url
                        logger.info(f"Selected format: {fmt.get('format_id')}, resolution: {fmt.get('width')}x{fmt.get('height')}")
                        break
            
            if not stream_url and 'requested_formats' in info:
                for fmt in info['requested_formats']:
                    if fmt.get('vcodec') != 'none':
                        stream_url = fmt.get('url')
                        if stream_url:
                            break
            
            if not stream_url:
                stream_url = info.get('url')
            
            if not stream_url:
                raise ValueError("Could not extract stream URL from YouTube")
            
            # Get video dimensions
            width = info.get('width', 640)
            height = info.get('height', 480)
            if not width or not height:
                formats = info.get('formats', [])
                for fmt in formats:
                    if fmt.get('vcodec') != 'none':
                        width = fmt.get('width', 640)
                        height = fmt.get('height', 480)
                        break
            self.width = width or 640
            self.height = height or 480
            self.frame_size = self.width * self.height * 3
            
            # Use ffmpeg to decode stream
            logger.info(f"Starting ffmpeg process for stream: {self.width}x{self.height}")
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', stream_url,
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{self.width}x{self.height}',  # Force output size
                '-r', str(FPS_LIMIT),
                '-loglevel', 'error',
                '-'
            ]
            
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10**8
            )
            
            time.sleep(0.1)
            
            if not self.ffmpeg_process or self.ffmpeg_process.poll() is not None:
                raise ValueError(f"Cannot start ffmpeg for YouTube stream: {youtube_url}")
            
            logger.info(f"ffmpeg process started successfully")
                
        except ImportError:
            logger.error("yt-dlp not installed. Install with: pip install yt-dlp")
            raise ValueError("yt-dlp is required for YouTube streaming")
        except FileNotFoundError:
            logger.error("ffmpeg not found. Install with: apt-get install ffmpeg or brew install ffmpeg")
            raise ValueError("ffmpeg is required for YouTube streaming")
        except Exception as e:
            logger.error(f"Error opening YouTube stream: {e}")
            if self.ffmpeg_process:
                self.ffmpeg_process.terminate()
            raise ValueError(f"Failed to open YouTube stream: {str(e)}")

    async def get_frame(self) -> np.ndarray:
        """Get next frame from YouTube stream."""
        try:
            # Read exact frame size
            raw_frame = self.ffmpeg_process.stdout.read(self.frame_size)
            
            if len(raw_frame) != self.frame_size:
                # Frame incomplete or stream ended
                if self.last_annotated_frame is not None:
                    frame = self.last_annotated_frame.copy()
                else:
                    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            else:
                # Convert bytes to numpy array
                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                # Reshape to (height, width, channels) - BGR format
                frame = frame.reshape((self.height, self.width, 3))
                # Copy to make it writable (frombuffer creates readonly array)
                frame = frame.copy()
                
                # Verify frame shape
                if frame.shape != (self.height, self.width, 3):
                    logger.error(f"Frame shape mismatch: expected ({self.height}, {self.width}, 3), got {frame.shape}")
                    if self.last_annotated_frame is not None:
                        frame = self.last_annotated_frame.copy()
                    else:
                        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        except Exception as e:
            logger.error(f"Error reading frame from ffmpeg: {e}")
            if self.last_annotated_frame is not None:
                frame = self.last_annotated_frame.copy()
            else:
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Process all frames through AI (no skipping)
        frame = await self._annotate_frame(frame)
        return frame

    async def _annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Annotate frame with YOLO detections from BentoML service."""
        async with aiohttp.ClientSession() as session:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_bytes = io.BytesIO()
            pil_image.save(img_bytes, format="JPEG")
            img_bytes.seek(0)

            data = aiohttp.FormData()
            data.add_field("image", img_bytes, filename="image.jpg", content_type="image/jpeg")

            async with session.post(f"{BENTO_ENDPOINT_URL}/predict", data=data) as resp:
                if resp.status != 200:
                    logger.error(f"BentoML service error: {resp.status}")
                    return frame

                result = await resp.json()
                detections = result.get("detections", [])

                for det in detections:
                    xyxy = det.get("xyxy", [])
                    confidence = det.get("confidence", 0.0)
                    class_id = det.get("class_id", 0)
                    class_name = det.get("class_name", str(class_id))

                    if len(xyxy) == 4:
                        x1, y1, x2, y2 = map(int, xyxy)
                        color = self._get_color(class_id)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{class_name} {int(confidence * 100)}%"
                        cv2.putText(
                            frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                        )

        return frame

    def _get_color(self, class_id: int) -> tuple:
        """Get color for class ID."""
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
        return colors[class_id % len(colors)]

    def cleanup(self):
        """Cleanup ffmpeg process."""
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=5)
            except Exception:
                try:
                    self.ffmpeg_process.kill()
                except Exception:
                    pass


async def websocket_youtube_stream(request: web.Request):
    """WebSocket endpoint for YouTube stream with YOLO detection."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    youtube_url = None
    streamer = None

    try:
        # Receive YouTube URL from client
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get("type") == "start":
                    youtube_url = data.get("youtube_url", "")
                    if not youtube_url:
                        await ws.send_str(json.dumps({"type": "error", "message": "YouTube URL is required"}))
                        break
                    
                    if "youtube.com" not in youtube_url and "youtu.be" not in youtube_url:
                        await ws.send_str(json.dumps({"type": "error", "message": "Invalid YouTube URL"}))
                        break

                    logger.info(f"Starting YouTube stream for: {youtube_url}")
                    streamer = YouTubeStreamer(youtube_url)
                    await ws.send_str(json.dumps({
                        "type": "ready",
                        "width": streamer.width,
                        "height": streamer.height
                    }))
                    break
                elif data.get("type") == "stop":
                    break
            elif msg.type == WSMsgType.ERROR:
                logger.error(f"WebSocket error: {ws.exception()}")
                break

        # Stream frames
        if streamer:
            frame_interval = 1.0 / FPS_LIMIT
            last_frame_time = time.time()
            
            while not ws.closed:
                current_time = time.time()
                if current_time - last_frame_time >= frame_interval:
                    frame = await streamer.get_frame()
                    
                    # Verify frame shape before encoding
                    if frame.shape != (streamer.height, streamer.width, 3):
                        logger.warning(f"Frame shape mismatch: expected ({streamer.height}, {streamer.width}, 3), got {frame.shape}")
                        # Skip this frame
                        await asyncio.sleep(0.01)
                        continue
                    
                    # Encode frame as JPEG
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if buffer is None or len(buffer) == 0:
                        logger.warning("Failed to encode frame as JPEG")
                        await asyncio.sleep(0.01)
                        continue
                    
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    await ws.send_str(json.dumps({
                        "type": "frame",
                        "data": frame_base64
                    }))
                    
                    last_frame_time = current_time
                else:
                    await asyncio.sleep(0.01)  # Small sleep to avoid busy waiting

    except Exception as e:
        logger.error(f"Error in WebSocket stream: {e}")
        import traceback
        logger.error(traceback.format_exc())
        if not ws.closed:
            await ws.send_str(json.dumps({"type": "error", "message": str(e)}))
    finally:
        if streamer:
            streamer.cleanup()
        await ws.close()
        logger.info("WebSocket connection closed")

    return ws


async def upload_image(request: web.Request) -> web.Response:
    """Upload image and return annotated image with bounding boxes."""
    data = await request.post()
    if "image" not in data:
        raise web.HTTPBadRequest(text="No image provided")

    image_file = data["image"]
    if not hasattr(image_file, "file"):
        raise web.HTTPBadRequest(text="Invalid image file")

    image_bytes = image_file.file.read()
    pil_image = Image.open(io.BytesIO(image_bytes))
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    frame = np.array(pil_image)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Annotate with YOLO
    async with aiohttp.ClientSession() as session:
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        form_data = aiohttp.FormData()
        form_data.add_field("image", img_bytes, filename="image.jpg", content_type="image/jpeg")

        async with session.post(f"{BENTO_ENDPOINT_URL}/predict", data=form_data) as resp:
            if resp.status != 200:
                logger.error(f"BentoML service error: {resp.status}")
                raise web.HTTPInternalServerError(text="BentoML service error")

            result = await resp.json()
            detections = result.get("detections", [])

            for det in detections:
                xyxy = det.get("xyxy", [])
                confidence = det.get("confidence", 0.0)
                class_id = det.get("class_id", 0)
                class_name = det.get("class_name", str(class_id))

                if len(xyxy) == 4:
                    x1, y1, x2, y2 = map(int, xyxy)
                    color = (0, 255, 0) if class_id == 0 else (255, 0, 0)
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name} {int(confidence * 100)}%"
                    cv2.putText(
                        frame_bgr, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )

    annotated_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    output_bytes = io.BytesIO()
    annotated_pil.save(output_bytes, format="JPEG")
    output_bytes.seek(0)

    return web.Response(
        body=output_bytes.read(),
        content_type="image/jpeg"
    )


async def cors_middleware(app, handler):
    """Simple CORS middleware."""
    async def middleware_handler(request):
        if request.method == "OPTIONS":
            response = web.Response()
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            return response
        
        response = await handler(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response
    return middleware_handler


def create_app() -> web.Application:
    """Create and configure aiohttp application."""
    app = web.Application(middlewares=[cors_middleware])
    app.router.add_get("/ws/youtube", websocket_youtube_stream)
    app.router.add_post("/api/v1/upload", upload_image)
    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLO YouTube Streaming API via WebSocket")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server")
    parser.add_argument("--port", type=int, default=8005, help="Port for HTTP server")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    app = create_app()
    web.run_app(app, host=args.host, port=args.port)
