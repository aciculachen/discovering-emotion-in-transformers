#!/bin/bash
# 推送整個 submission 到 server (不含 submission 子目錄)
# Server 結構: /home/kevin/Final_project_MI/{src,viz,results,data,run_all.sh}

LOCAL_DIR="/Users/acicula/Documents/2025 AU/CSE 6521 /Final_project_MI/submission"
REMOTE_HOST="kevin@100.73.196.77"
REMOTE_DIR="/home/kevin/Final_project_MI"

echo "🚀 Pushing to server..."

# Push src/
rsync -avz --exclude='*.pyc' --exclude='__pycache__' \
    "${LOCAL_DIR}/src/" \
    "${REMOTE_HOST}:${REMOTE_DIR}/src/"

# Push viz/
rsync -avz --exclude='*.pyc' --exclude='__pycache__' \
    "${LOCAL_DIR}/viz/" \
    "${REMOTE_HOST}:${REMOTE_DIR}/viz/"

# Push data/ (if exists)
if [ -d "${LOCAL_DIR}/data" ]; then
    rsync -avz "${LOCAL_DIR}/data/" "${REMOTE_HOST}:${REMOTE_DIR}/data/"
fi

# Push shell scripts
rsync -avz "${LOCAL_DIR}/run_all.sh" "${REMOTE_HOST}:${REMOTE_DIR}/"
rsync -avz "${LOCAL_DIR}/run_all_exp.sh" "${REMOTE_HOST}:${REMOTE_DIR}/"
rsync -avz "${LOCAL_DIR}/run_all_viz.sh" "${REMOTE_HOST}:${REMOTE_DIR}/"

# Push results/ (CSV files only, not figures)
rsync -avz --include='*.csv' --exclude='figures/' \
    "${LOCAL_DIR}/results/" \
    "${REMOTE_HOST}:${REMOTE_DIR}/results/"

echo "✅ Push complete!"
echo "Server path: ${REMOTE_DIR}"
