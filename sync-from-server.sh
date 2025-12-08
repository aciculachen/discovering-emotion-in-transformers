#!/bin/bash
# 從 server 拉回結果
# Server 結構: /home/kevin/Final_project_MI/{src,viz,results,data,run_all.sh}

LOCAL_DIR="/Users/acicula/Documents/2025 AU/CSE 6521 /Final_project_MI/submission"
REMOTE_HOST="kevin@100.73.196.77"
REMOTE_DIR="/home/kevin/Final_project_MI"

echo "📥 Pulling results from server..."

# Pull results/ (all files including figures)
rsync -avz \
    "${REMOTE_HOST}:${REMOTE_DIR}/results/" \
    "${LOCAL_DIR}/results/"

echo "✅ Pull complete!"
echo "Local path: ${LOCAL_DIR}/results/"
