from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch
from request import ModelRequest
import librosa
import argparse

class Model():
    def __new__(cls, context):
        cls.context = context
        if not hasattr(cls, 'instance'):
            cls.instance = super(Model, cls).__new__(cls)
        
        model_name = "facebook/mms-1b-all"
        target_lang = "ory"
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        model.load_adapter(target_lang)
        cls.model = model 
        processor =  AutoProcessor.from_pretrained(model_name)
        processor.tokenizer.set_target_lang(target_lang)
        cls.processor =  processor
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model.to(cls.device)
        return cls.instance


    async def inference(self,  request: ModelRequest):
        wav_file = request.wav_file
        ory_sample, sr = librosa.load(wav_file, sr=16000)

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--extra-infer-args",
            help="Extra arguments to pass to the model's inference function.",
        )
        args = parser.parse_args()
        decoding_cmds = """
        decoding.type=kenlm
        decoding.beam=500
        decoding.beamsizetoken=50
        decoding.lmweight=2.69
        decoding.wordscore=2.8
        decoding.lmpath= kenlmFiles/new_5gram_test.bin
        decoding.lexicon= kenlmFiles/new_lexicon_test.txt
        """.replace("\n", " ")
        args.extra_infer_args["decoding_cmds"] = decoding_cmds

        inputs = self.processor(ory_sample, sampling_rate=16_000, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, **args.extra_infer_args).logits
        
        ids = torch.argmax(outputs, dim=-1)[0]
        transcription = self.processor.decode(ids)
        
        return transcription