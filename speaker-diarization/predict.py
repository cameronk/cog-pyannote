from typing import Any, List 

import numpy as np
from cog import BasePredictor, Input, Path, BaseModel
from pyannote.audio import Pipeline
import torch
import logging
import ffmpeg

class TurnWithSpeaker(BaseModel):
    start : float
    end : float
    speaker: str

# https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/pipelines/speaker_diarization.py#L56
class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        
        # Check device
        device_count = torch.cuda.device_count()
        logging.info("Available GPUs: %s" % device_count)
        if device_count == 0: 
            raise Exception("GPU unavailable, device count is %s" % device_count)

        # Load model
        self.pipeline = Pipeline.from_pretrained("config.yaml")
        logging.info("Completed setup")
        pass

    def hook(self, name : str, step_artefact : Any, file : Any) -> None:
        """Called during pyannote pipeline execution"""
        logging.info("Hook: %s %s" % (name, step_artefact))
        pass

    def predict(
        self,
        audio: Path = Input(description="Audio to diarize. Accepts any audio file type convertible to .wav by ffmpeg."),
        num_speakers: int = Input(description="Number of speakers if known in advance", default=None),
        min_speakers: int = Input(description="Lower bound on number of speakers", default=None),
        max_speakers: int = Input(description="Upper bound on number of speakers", default=None),
    ) -> List[TurnWithSpeaker]:
        # Convert to wav if necessary
        audio_path = audio

        if audio.suffix != ".wav":
            logging.info("Converting audio to wav")
            audio_path = audio.with_suffix(".wav")
            try: 
                ffmpeg.input(audio).output(audio_path).run()
            except Exception as e:
                logging.exception(e)
                raise e

        # https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/pipelines/speaker_diarization.py#L422
        diarization = self.pipeline(
            audio_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            hook=self.hook
        )

        logging.info("Diarization complete")

        results = [
            # https://github.com/pyannote/pyannote-core/blob/b5df979d1215012260717d35a48113195131ddad/pyannote/core/annotation.py#L265
            TurnWithSpeaker(
                start=segment.start,
                end=segment.end,
                speaker=speaker
            ) for segment, _, speaker in diarization.itertracks(yield_label=True)
        ]

        logging.info("Found %s results" % len(results))

        return results


