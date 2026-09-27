#!/usr/bin/env bash
# Simple re-streamer: RTSP -> local HTTP (MPEG-TS)
# Usage: ./restream_rtsp.sh "rtsp://user:pass@192.168.1.50:554/stream1" 8090

INPUT="$1"
PORT="${2:-8090}"

if [ -z "$INPUT" ]; then
  echo "Usage: $0 \"rtsp://user:pass@IP:PORT/path\" [port]"
  exit 1
fi

echo "Re-streaming $INPUT -> http://0.0.0.0:$PORT/feed1"
ffmpeg -rtsp_transport tcp -i "$INPUT" -f mpegts "http://0.0.0.0:$PORT/feed1"
