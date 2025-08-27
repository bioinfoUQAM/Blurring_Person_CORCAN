#!/bin/bash
# Define output name variable

#SBATCH --account=def-banire           # Conta do Compute Canada
#SBATCH --gpus-per-node=1              # Solicita 1 GPU (P100 ou outra disponível)
#SBATCH --constraint=v100l              # Garante que a GPU seja uma P100
#SBATCH --cpus-per-task=2              # Ajusta o número de CPUs
#SBATCH --mem=10G                     # Ajusta a memória
#SBATCH --time=29:50:00                # Define o tempo limite
#SBATCH --mail-user=marcelo_de_araujo.voncarlos@uqam.ca  # Email para notificações
#SBATCH --mail-type=ALL                # Notificação em START, END, FAIL
#SBATCH --output=%x-%j.out             # Nome do arquivo de saída baseado no job name e ID

# ==== VARIÁVEIS ====
export DATA_YAML="/home/vonzin/scratch/Blurring2025/Dataset_4_head_person/data.yaml"
export PROJECT_PATH="/home/vonzin/scratch/Blurring2025/Modelos_treinados"
export RUN_NAME="fromscract_newdata_350epoch_Dataset_4_head_person_2"
export YOLOV8_PATH="/home/vonzin/scratch/Blurring2025/CORCAN_blurring/yolov8x.pt"
export EPOCHS=350

#module load python/3.11.5 cuda/12.2  
source ~/envs/my_env/bin/activate   
module restore my_enviroment

cd /home/vonzin/scratch/Blurring2025/CORCAN_blurring

  
python train_yolov8.py \
  --model ${YOLOV8_PATH} \
  --data "${DATA_YAML}" \
  --epochs ${EPOCHS} \
  --imgsz 960 \
  --batch 4 \
  --optimizer SGD \
  --verbose \
  --patience 20 \
  --name "${RUN_NAME}" \
  --project "${PROJECT_PATH}" \
  --device 0

# ==== LOG DE FIM ====
echo "Treinamento YOLOv8 finalizado em $(date)"
