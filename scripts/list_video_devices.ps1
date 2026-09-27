# List DirectShow video devices using ffmpeg
# Usage: .\list_video_devices.ps1

$ffmpeg = "ffmpeg"  # must be in PATH
Write-Host "Listing DirectShow devices via ffmpeg..."

try {
    & $ffmpeg -list_devices true -f dshow -i dummy 2>&1 | ForEach-Object { Write-Host $_ }
} catch {
    Write-Host "Error: ffmpeg not found or failed. Install ffmpeg and ensure it's in PATH."
}
