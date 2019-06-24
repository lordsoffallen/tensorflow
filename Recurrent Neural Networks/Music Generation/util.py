from tensorflow.python.keras.utils import plot_model
from IPython.display import Audio, Image
from IPython import display
import matplotlib.pyplot as plt
import time
import os
import re


def _extract_song_snippet(generated):
    pattern = '\n\n(.*?)\n\n'
    found = re.findall(pattern, generated, flags=re.DOTALL)
    songs = [song for song in found]
    print("Found {} possible songs in generated texts".format(len(songs)))
    return songs


def _save_song_to_abc(song, filename="gen_song"):
    save_name = "{}.abc".format(filename)
    with open(save_name, "w") as f:
        f.write(song)
    return filename


def _abc2wav(abc, wav='data/abc2wav'):
    return os.system("bash {} {}".format(wav, abc))


def _play_wav(wav_file):
    return Audio(wav_file)


def play_generated_song(generated):
    """ Converts the generated abc to wav and then plays it

    Parameters
    ----------
    generated:
        Generated song in abc notation
    """

    songs = _extract_song_snippet(generated)
    if len(songs) == 0:
        print("No valid songs found in generated text. Try training the model longer or "
              "increasing the amount of generated music to ensure complete songs are generated!")

    for song in songs:
        basename = _save_song_to_abc(song)
        ret = _abc2wav(basename + '.abc')

        if ret == 0: # did not succeed
            return _play_wav(basename + '.wav')

    print("None of the songs were valid, try training longer to improve syntax.")


def display_model(model):
    plot_model(model, to_file='model.png', show_shapes=True)
    return Image('model.png')


class PeriodicPlotter:
    def __init__(self, sec, xlabel='', ylabel='', scale=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.sec = sec
        self.scale = scale
        self.tic = time.time()

    def plot(self, data):
        if time.time() - self.tic > self.sec:
            plt.cla()

            if self.scale is None:
                plt.plot(data)
            elif self.scale == 'semilogx':
                plt.semilogx(data)
            elif self.scale == 'semilogy':
                plt.semilogy(data)
            elif self.scale == 'loglog':
                plt.loglog(data)
            else:
                raise ValueError("unrecognized parameter scale {}".format(self.scale))

            plt.xlabel(self.xlabel)
            plt.ylabel(self.ylabel)
            display.clear_output(wait=True)
            display.display(plt.gcf())