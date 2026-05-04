# Reasoning Distillation on Google LLMs
## Ethan Hao, Darren Kao
## Yale University, New Haven, CT
## {ethan.hao, darren.kao}@yale.edu

All code used in this project is attached in this GitHub.

To run evaluation on the baseline version of Gemini:
1. In the terminal, add a Google API key in the environment variable GOOGLE_API_KEY, and add an OpenAI API key in the environment variable OPENAI_API_KEY.
2. Install any dependencies required.
3. Run the following line in the terminal:
`python run_gemini_predictions.py --google_api_key $GOOGLE_API_KEY --dataset cais/hle --num_workers 2`
4. Run the following line in the terminal:
`python run_judge_results.py --openai_api_key $OPENAI_API_KEY --model gemini-2.5-flash-lite --predictions gemini-2.5-flash-lite_results.json`

To run evaluation on the baseline version of PaliGemma:
1. In the terminal, add a HuggingFace API key in the environment variable HF_API_KEY, and add an OpenAI API key in the environment variable OPENAI_API_KEY.
2. Install any dependencies required.
3. Run the following line in the terminal:
`python run_paligemma_predictions.py --sample 250 --output base_paligemma_results.json --hf-token #HF_API_KEY --seed 40`
4. Run the following line in the terminal:
`python run_judge_results.py --openai_api_key $OPENAI_API_KEY --model base_paligemma --predictions base_paligemma_results.json`

To run the fine-tuning training of PaliGemma:
1. Open the PaliGemma_Fine_Tuning.ipynb notebook, and add hle_gemini-2.5-flash-lite.json underneath its files.
2. Run each cell in order.
3. Download the distilled_adapter folder.

To run evaluation on the fine-tuned version of PaliGemma:
1. Open the run_finetuned_paligemma_predictions.ipynb notebook, and add both hle_gemini-2.5-flash-lite.json and a distilled_adapter folder underneath its files.
2. Run each cell in order.
3. Download the finetuned_paligemma_results.json file.
4. In the terminal, add an OpenAI API key in the environment variable OPENAI_API_KEY.
5. Install any dependencies required.
6. Run the following line in the terminal:
`python run_judge_results.py --openai_api_key $OPENAI_API_KEY --model finetuned_paligemma --predictions finetuned_paligemma_results.json`