#!/bin/bash

create_asm_structure() {
    read -p "Enter directory path (or press Enter for current directory): " directory
    read -p "Enter file name (with .asm extension): " file_name

    directory="${directory:-$(pwd)}"

    if [[ ! "$file_name" == *.asm ]]; then
        file_name="$file_name.asm"
    fi

    asm_code="section .text
global _start

_start:
    ; Your assembly code here
"

    mkdir -p "$directory"
    echo "$asm_code" > "$directory/$file_name"

    echo "Assembly structure created in $directory/$file_name"
}

if [ "$1" == "asm_start" ]; then
    create_asm_structure
else
    echo "Unknown command. Use 'asm_start' to create an assembly structure."
fi