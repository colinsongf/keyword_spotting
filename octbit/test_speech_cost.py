# coding=utf-8
import os
import datetime
import time
from speech.metric.wer import wer as wer_calculation

from speech_pb2 import CLMInputUnit, CLMInput, SpeechInput
from asr_component.speech_service import SpeechService
import sys
from plugins.octbit import octbit_mat_mul

MAX_ERROR_RATE = 10000
VER = sys.version[0]

if __name__ == '__main__':
    import argparse

    from asr_component.service_parser import parser, pinyin_dict
    parser.add_argument("-tc", "--circle_number", default=10, type=int)
    FLAGS = parser.parse_args()
    speech_service = SpeechService()
    speech_service.load_graph(FLAGS.speech_graph_pb_path,
                              py_dict=FLAGS.speech_pinyin_dict)

    id_ = "octbit.wav"
    if VER == '2':
        audio_data = open(id_, 'r').read()
    else:
        audio_data = open(id_, "r", encoding="latin-1").read()
    sinputs = []
    sinputs.append(audio_data)
    start_time = time.time()
    context = ""
    for i in range(FLAGS.circle_number):
        for speech_output in speech_service.SpeechRecognize([
                SpeechInput(sinput=sinputs[0],
                            error_code=-1,
                            contactJson="")],
                context):
            pass
    print(time.time() - start_time)
