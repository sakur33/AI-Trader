$ErrorActionPreference = 'SilentlyContinue'

Write-Output "Installing Docker"
if (Test-Path -Path "C:\'Program Files'\Docker\")
    {
        $dockerv = Invoke-Expression "docker --version"
        Write-Output "Docker already installed: $dockerv"
    }
    else 
    {
        Invoke-WebRequest -UseBasicParsing -Uri "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe" -OutFile "./temp/docker.exe";
        Start-Process ".\temp\docker.exe"
    }
    Invoke-Expression "docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=1234 --name tick_db timescale/timescaledb-ha:latest-pg14"        

Write-Output "Installing DBeaver"
if (Test-Path -Path "C:\Program Files\DBeaver")
    {
        Write-Output "DBeaver already installed"
    }
    else 
    {
        Invoke-WebRequest -UseBasicParsing -Uri "https://dbeaver.io/files/dbeaver-ce-latest-x86_64-setup.exe" -OutFile "./temp/dbeaver.exe";
        Start-Process "./temp/dbeaver.exe"
    }
    Invoke-Expression "C:\Program Files\DBeaver\dbeaver.exe"
