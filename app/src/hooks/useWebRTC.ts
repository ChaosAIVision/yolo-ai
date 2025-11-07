/// <reference types="vite/client" />
import { useRef, useCallback, useEffect } from 'react';

const STUN_SERVER = import.meta.env.VITE_STUN_SERVER || 'stun:stun.l.google.com:19302';
const TURN_URL = import.meta.env.VITE_TURN_URL || '';
const TURN_USERNAME = import.meta.env.VITE_TURN_USERNAME || '';
const TURN_CREDENTIAL = import.meta.env.VITE_TURN_CREDENTIAL || '';

export interface UseWebRTCReturn {
  createPeerConnection: () => RTCPeerConnection;
  createOffer: (pc: RTCPeerConnection) => Promise<RTCSessionDescriptionInit | null>;
  setAnswer: (pc: RTCPeerConnection, answer: RTCSessionDescriptionInit) => Promise<void>;
  cleanup: (pc: RTCPeerConnection | null) => void;
}

export function useWebRTC(): UseWebRTCReturn {
  const createPeerConnection = useCallback(() => {
    const iceServers: RTCIceServer[] = [{ urls: STUN_SERVER }];
    if (TURN_URL) {
      iceServers.push({
        urls: TURN_URL,
        username: TURN_USERNAME || undefined,
        credential: TURN_CREDENTIAL || undefined,
      } as RTCIceServer);
    }
    const pc = new RTCPeerConnection({
      iceServers,
    });
    return pc;
  }, []);

  const createOffer = useCallback(async (pc: RTCPeerConnection) => {
    try {
      // Ensure the offer includes a video m-line so the server can send a track
      pc.addTransceiver('video', { direction: 'recvonly' })

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      // Wait for ICE gathering to complete so SDP includes candidates (non-trickle)
      await new Promise<void>((resolve) => {
        if (pc.iceGatheringState === 'complete') {
          resolve();
        } else {
          const checkState = () => {
            if (pc.iceGatheringState === 'complete') {
              pc.removeEventListener('icegatheringstatechange', checkState);
              resolve();
            }
          };
          pc.addEventListener('icegatheringstatechange', checkState);
        }
      });

      // Return the finalized local description with ICE candidates embedded
      return pc.localDescription;
    } catch (error) {
      console.error('Error creating offer:', error);
      return null;
    }
  }, []);

  const setAnswer = useCallback(async (pc: RTCPeerConnection, answer: RTCSessionDescriptionInit) => {
    try {
      await pc.setRemoteDescription(new RTCSessionDescription(answer));
    } catch (error) {
      console.error('Error setting answer:', error);
      throw error;
    }
  }, []);

  const cleanup = useCallback((pc: RTCPeerConnection | null) => {
    if (pc) {
      pc.close();
    }
  }, []);

  return {
    createPeerConnection,
    createOffer,
    setAnswer,
    cleanup,
  };
}

