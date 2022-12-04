from typing import Any, List 

import numpy as np
from cog import BasePredictor, Input, Path, BaseModel
from pyannote.audio import Pipeline
import torch
import logging

class TurnWithSpeaker(BaseModel):
  start : int
  end : int
  speaker: int

# https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/pipelines/speaker_diarization.py#L56
class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
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
      device = "cuda:0" if torch.cuda.is_available() else "cpu"
      logging.info("[cog/speaker-diarization] using device: %s" % device)

      if device != "cuda:0":
        raise "GPU not available"

      # https://github.com/pyannote/pyannote-audio/blob/f700d6ea8dedd42e7c822c3b44b46a952e62a585/pyannote/audio/core/pipeline.py#L46
      self.pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1",
        use_auth_token=auth_token,
        device=device
      )

      logging.info("[cog/speaker-diarization] loaded pipeline")

      if audio.suffix != "wav":
        raise "Input file must be wav"

      diarization = self.pipeline(
        audio,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers
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


