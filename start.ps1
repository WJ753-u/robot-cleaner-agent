param(
    [int]$ApiPort = 8000,
    [int]$WebPort = 8501,
    [switch]$ApiReload
)

$scriptPath = Join-Path $PSScriptRoot "scripts\start_services.py"
$argsList = @($scriptPath, "--api-port", $ApiPort, "--web-port", $WebPort)

if ($ApiReload) {
    $argsList += "--api-reload"
}

py @argsList
