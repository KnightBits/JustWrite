@echo off
setlocal enabledelayedexpansion

if "%1"=="asm_start" (
    call :create_asm_structure
) else (
    echo Unknown command. Use 'asm_start' to create an assembly structure.
)

exit /b 0

:create_asm_structure
set /p directory="Enter directory path (or press Enter for current directory): "
set /p file_name="Enter file name (with .asm extension): "

if not defined directory (
    set "directory=%cd%"
)

if not "!file_name:~-4!"==".asm" (
    set "file_name=!file_name!.asm"
)

set "asm_code=section .text
global _start

_start:
    ; Your assembly code here
"

mkdir "%directory%" 2>nul
echo %asm_code% > "%directory%\%file_name%"

echo Assembly structure created in %directory%\%file_name%
goto :eof