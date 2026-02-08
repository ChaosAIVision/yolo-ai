#!/bin/bash
# Test script for API endpoints

echo "=========================================="
echo "Testing YOLO AI API Endpoints"
echo "=========================================="
echo ""

API_URL="http://localhost:6000"

echo "1. Health Check:"
curl -s "${API_URL}/health" | jq . || curl -s "${API_URL}/health"
echo ""
echo ""

echo "2. API Health Check:"
curl -s "${API_URL}/api/health" | jq . || curl -s "${API_URL}/api/health"
echo ""
echo ""

echo "3. Upload Endpoint (without image - should return error):"
curl -s -X POST "${API_URL}/api/v1/upload" | head -100
echo ""
echo ""

echo "4. Test with CORS headers:"
curl -s -H "Origin: http://localhost:7070" \
     -H "Access-Control-Request-Method: POST" \
     -X OPTIONS "${API_URL}/api/v1/upload" \
     -v 2>&1 | grep -E "HTTP|Access-Control"
echo ""
echo ""

echo "=========================================="
echo "API Test Complete"
echo "=========================================="
echo ""
echo "If all tests pass, API is working correctly!"
echo "Access API at: ${API_URL}"

