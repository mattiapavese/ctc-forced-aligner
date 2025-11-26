import torch, torchaudio
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
from .text_utils import preprocess_text, postprocess_results, SAMPLING_FREQ
from .alignment_utils import get_spans, get_alignments, generate_emissions_batch
from typing import TypedDict

class Word(TypedDict):
    start:float
    end:float
    text:str
    score:float

def align_batch(
    model:Wav2Vec2ForCTC, 
    tokenizer:Wav2Vec2Tokenizer,
    audios:list[torch.Tensor]|torch.Tensor, 
    sr:int,
    texts:list[str]|str, 
    batch_size:int=16, language:str|None=None
)->list[tuple[list[Word], bool]]:
    
    """Returns a list of tuples (list[Word], ok),
    where ok is a bool indicating whether the alignment process
    terminated successfully for a particular entry or not."""
    
    if isinstance(audios, torch.Tensor):
        audios=[audios]
    if isinstance(texts, str):
        texts=[texts]
    assert len(audios)==len(texts), "Number of audios and texts to align must be equal."

    if sr!=SAMPLING_FREQ:
       audios = [ torchaudio.functional.resample(a, sr, SAMPLING_FREQ) for a in audios]

    emissions, _ = generate_emissions_batch(
        model, tensors=audios, batch_size=batch_size
    )
    
    word_timestamps_coll=[]

    # could gain some (relatively little) speed up 
    # by executing this in parallel
    # using multiprocessing
    for emission,text in zip(emissions, texts):
        try:
            tokens_starred, text_starred = preprocess_text(
                text, romanize=True, language=language,
            )

            segments, scores, blank_token = get_alignments(
                emission, tokens_starred, tokenizer,
            ) 
            
            spans = get_spans(
                tokens_starred, segments, blank_token
            )

            word_timestamps:list[Word] = postprocess_results(
                text_starred, spans, scores
            )
            
            word_timestamps_coll.append((word_timestamps, True))
        except Exception as err:
            print(f"[ERROR] Failed to align '{text}': {err}\nAppending empty list[Word].")
            print("...")
            word_timestamps_coll.append(([], False))
    
    return word_timestamps_coll