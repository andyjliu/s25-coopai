source /home/andyliu/Miniconda3/etc/profile.d/conda.sh
export CONDA_ALWAYS_YES="true"
conda activate coopai
cd /home/andyliu/s25-coopai

declare -a strategies=("always_defect" "cooperative" "defensive" "tit_for_tat")

for s1 in "${strategies[@]}"; do
    for s2 in "${strategies[@]}"; do
        if [[ "$s1" < "$s2" ]]; then
            python src/prisoners_dilemma.py --agent1 $s1 --agent2 $s2 --predict --output pd_gpt_explicit.csv
            python src/prisoners_dilemma.py --agent1 $s1 --agent2 $s2 --output pd_gpt_inferred.csv
        fi
    done
done