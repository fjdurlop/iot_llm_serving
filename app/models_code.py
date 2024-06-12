""" models

This module defines the classes of each model used in the API.

To add a new model:
    1. Add Models_names
    2. Add ML_task
    3. Create new class:
        def class NewModel(Model):
    4. Create schema in schemas
    5. Add endpoint in api
    
ToDo:
- Add max_new_tokens parameter

Models:
    - codet5-base
    - codet5p-220
    - codegen-350-mono
    - gpt-neo-125m
    - codeparrot-small
    - pythia-410m
    
"""

# External
#from codecarbon import track_emissions
from enum import Enum

# Required to run CNN model
import numpy as np
import random
#from torch.nn import functional as F
#import torch

# [t5, codet5p_220m]
#from transformers import T5ForConditionalGeneration
# [codegen]
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSeq2SeqLM
# [pythia-70m]
#from transformers import GPTNeoXForCausalLM, AutoTokenizer

# gptneo
#from transformers import pipeline

# onnxruntime
#from optimum.onnxruntime import ORTModelForMaskedLM, ORTModelForSeq2SeqLM, ORTModelForCausalLM
#from optimum.intel import OVModelForSeq2SeqLM, OVModelForCausalLM


# metrics

# Constants
RESULTS_DIR = 'results/'

models = [ 'pythia-410m'] # bloom, pythia

runtime_engines = ['onnx','ov','torchscript']

class ML_task(Enum):
    MLM = 1 # Masked Language Modeling
    TRANSLATION = 2
    CV = 3 # Computer Vision
    CODE = 4

class models_names(Enum):
    Codet5_base = 1
    Codet5p_220m = 2
    CodeGen_350m_mono = 3
    GPT_Neo_125m = 4
    CodeParrot_small = 5
    Pythia_410m = 6
    # BERT = 1
    # T5 = 2
    # CNN = 4
    # Pythia_70m = 5
    # Codet5p_220m = 6    
    

class Model:
    """
    Creates a default model
    """
    def __init__(self, model_name : models_names = None, ml_task : ML_task = None):
        self.name = model_name.name
        # Masked Language Modeling - MLM
        self.ml_task = ml_task.name
    
    def predict(self, user_input : str) -> dict:
        # Do prediction
        prediction = "Not defined yet "
        response = {
            "prediction" : prediction
        }
        return response
    
    
    def infer(self, text : str, model, tokenizer) -> str:
        """_summary_ Infer function to track

        Args:
            text (str): _description_
            model (_type_): _description_
            tokenizer (_type_): _description_

        Returns:
            str: _description_
        """
        
        #input_ids = tokenizer(text, return_tensors="pt").input_ids
        #outputs = model.generate(input_ids)
        #return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        response = None
        return response
        

#running
class Pythia_410mModel(Model):
    """_summary_ Creates a Pythia model. Inherits from Model()

    Args:
        Model (_type_): _description_
    """

    def __init__(self):
        super().__init__(models_names.Pythia_410m, ML_task.CODE)
        
    def predict(self, user_input: str, engine = None,):
        
        print(f'Runtime engine: {engine}')

        if engine not in runtime_engines:
            model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/pythia-410m",
            #revision="step3000",
            #cache_dir="./pythia-410m/step3000",
            )

            tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-410m",
            #revision="step3000",
            #cache_dir="./pythia-410m/step3000",
            )

        #@track_emissions(project_name = "pythia-410m", output_file = RESULTS_DIR + "emissions_pythia-410m.csv")
        #@decorator_to_use
        def infer( text: str, model, tokenizer, engine) -> str:
            if(engine != 'torchscript' ):
                # tokenize
                inputs = tokenizer(text, return_tensors="pt")
                # generate
                tokens = model.generate(**inputs)
                # decode
                prediction = tokenizer.decode(tokens[0])
                return prediction 
            
        response = {
            "prediction" : infer(user_input, model, tokenizer, engine)
        }
        
        return response
