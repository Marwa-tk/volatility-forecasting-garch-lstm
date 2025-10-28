#!/usr/bin/env bash
set -euo pipefail

yellow(){ printf "\033[33m%s\033[0m\n" "$*"; }
green(){  printf "\033[32m%s\033[0m\n" "$*"; }
red(){    printf "\033[31m%s\033[0m\n" "$*"; }

ensure_dir(){ mkdir -p "$1"; }

load_env(){
  if [ -f .env ]; then
    set -a
    . ./.env
    set +a
  else
    yellow " .env introuvable — j'utiliserai les valeurs par défaut (.env.example comme guide)."
  fi
}
