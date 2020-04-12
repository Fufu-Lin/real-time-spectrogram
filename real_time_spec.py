# -*- coding: utf-8 -*-
import os
import wave
import msvcrt
import pyaudio
import threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.mlab import window_hanning, specgram
from time import sleep



CHANNELS = 1
SAMPLES_PER_FRAME = 150
nfft = 512
overlap = 256
RATE = 16000
CHUNK = 2048
FORMAT = pyaudio.paInt16

im = None
RUUNING = True

input_wav = np.zeros([CHUNK])


p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)



def update_spec(i):


    data = input_wav


    pxx, _, _ = specgram(data, NFFT=nfft, Fs=RATE, noverlap=overlap, mode='magnitude')

    im_data = im.get_array()



    if i < SAMPLES_PER_FRAME:

        im_data = np.hstack((im_data, pxx))
        im.set_array(im_data)

    else:

        keep_block = 500#pxx.shape[1]*(SAMPLES_PER_FRAME - 1)
        im_data = np.delete(im_data, np.s_[:-keep_block], 1)
        im_data = np.hstack((im_data, pxx))
        im.set_array(im_data)


    return im,



def live_spec():

    global im, fig


    tmp_data = np.fromstring(stream.read(CHUNK), dtype=np.int16)

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 5))
    

    pxx, freq, t = specgram(tmp_data, NFFT=nfft, Fs=RATE, noverlap=overlap, mode='magnitude')
    extent = [t[0] ,t[-1]*3, freq[-1], freq[0]]

    plt.subplot(111)
    im = plt.imshow(pxx, cmap='hot', extent=extent, aspect='auto')
    plt.xticks([])

    plt.title('Real Time Spectrogram')
    plt.gca().invert_yaxis()

    anim = animation.FuncAnimation(fig, update_spec, blit=False, interval=10)

    try:

        plt.show()

    except KeyboardInterrupt:

        print('Plot closed')



def record():

    global input_wav, RUUNING

    while RUUNING:

        sleep(0.001)

        tmp = stream.read(CHUNK)

        input_wav = np.fromstring(tmp, dtype=np.int16)





def stop():

    global RUUNING

    while RUUNING:

        sleep(1)

        ch = msvcrt.getwch()

        if ch == 'q':

            RUUNING = False
            plt.close()

            break



def main():

    r_thread = threading.Thread(target=record, name='record')
    p_thread = threading.Thread(target=live_spec, name='live_spec')
    q_thread = threading.Thread(target=stop, name='stop')

    r_thread.start()
    p_thread.start()
    q_thread.start()

    r_thread.join()
    p_thread.join()
    q_thread.join()





if __name__ == '__main__':

    main()