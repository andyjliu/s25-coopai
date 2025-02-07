## s25 cooperative ai course project
```
conda create -n coopai python=3.10 -y
conda activate coopai
pip install vllm openai anthropic pandas numpy matplotlib sentence_transformers
conda env config vars set OPENAI_API_KEY=$secret
conda env config vars set ANTHROPIC_API_KEY=$secret
```
