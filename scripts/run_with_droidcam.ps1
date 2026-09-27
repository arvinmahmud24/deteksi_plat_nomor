<#
Simple helper to run main.py with a DroidCam Wi‑Fi URL.
Usage:
  .\run_with_droidcam.ps1 -Ip 192.168.1.20 -Path "/video"
#>
param(
    [Parameter(Mandatory=$true)][string]$Ip,
    [string]$Path = "/video",
    [string]$Name = "DroidCam-WiFi"
)

$camUrl = "http://$Ip:4747$Path"
Write-Host "Using DroidCam URL: $camUrl"

# Set env var for current session and run main.py
$env:CAM_URL = $camUrl
python main.py --source $env:CAM_URL --name $Name
