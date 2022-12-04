from typing import Any, List 

import numpy as np
from cog import BasePredictor, Input, Path, BaseModel
from pyannote.audio import Pipeline
import torch
import logging

logging.basicConfig(filename="predict.log", level=logging.DEBUG)

class TurnWithSpeaker(BaseModel):
  start : int
  end : int
  speaker: int

# https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/pipelines/speaker_diarization.py#L56
class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        pass

    def hook(self, name : str, step_artefact : Any, file : Any) -> None:
      logging.info("[cog/speaker-diarization] hook %s %s" % (name, step_artefact))

      if name == "on_predict":
        logging.info("[cog/speaker-diarization] on_predict %s" % step_artefact)
          
      pass

    # Define the arguments and types the model takes as input
    def predict(
      self,
      audio: Path = Input(description="Audio to diarize"),
      auth_token: str = Input(description="Huggingface auth_token used to load pretrained model"),
      num_speakers: int = Input(description="Number of speakers if known in advance", default=None),
      min_speakers: int = Input(description="Lower bound on number of speakers", default=None),
      max_speakers: int = Input(description="Upper bound on number of speakers", default=None),
    ) -> List[TurnWithSpeaker]:
      logging.info("[cog/speaker-diarization] running prediction")
      
      # Check device
      device_count = torch.cuda.device_count()
      logging.info("[cog/speaker-diarization] available gpus %s" % device_count)

      if device_count == 0: raise Exception("GPU unavailable")

      # https://github.com/pyannote/pyannote-audio/blob/f700d6ea8dedd42e7c822c3b44b46a952e62a585/pyannote/audio/core/pipeline.py#L46
      self.pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1",
        use_auth_token=auth_token
      )

      logging.info("[cog/speaker-diarization] loaded pipeline")

      if audio.suffix != ".wav":
        raise Exception("Expected extension .wav, got %s" % audio.suffix)

      # https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/pipelines/speaker_diarization.py#L422
      diarization = self.pipeline(
        audio,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        hook=self.hook
      )

      logging.info("[cog/speaker-diarization] diarized audio")

      results = [
        ({
          "start": turn.start,
          "end": turn.end,
          "speaker": speaker
        }) for turn, _, speaker in diarization.itertracks(yield_label=True)
      ]

      logging.info("[cog/speaker-diarization] found %s results" % len(results))

      return results


