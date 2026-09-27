param(
    [Parameter(Mandatory=$true)][string]$Input,
    [int]$Port = 8090
)

# PowerShell re-streamer: RTSP -> local HTTP MPEG-TS
# Usage: .\restream_rtsp.ps1 -Input "rtsp://user:pass@192.168.1.50:554/stream1" -Port 8090

$ffmpeg = "ffmpeg"
Write-Host "Re-streaming $Input -> http://0.0.0.0:$Port/feed1"

& $ffmpeg -rtsp_transport tcp -i $Input -f mpegts "http://0.0.0.0:$Port/feed1"
