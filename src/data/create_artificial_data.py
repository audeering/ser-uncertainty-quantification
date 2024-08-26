import audeer
import audformat
import audiofile
import auglib
import numpy as np


def create_white_noise_db(
        db_root='./'
):
    file_root = audeer.mkdir(db_root + '/white_noise/')
    file_duration = 8.0
    sampling_rate = 16_000
    db = audformat.Database("white-noise")

    gains = []
    files = []
    for gain in range(-50, 20, 2):
        transform = auglib.transform.WhiteNoiseGaussian(gain_db=gain)
        signal = np.zeros((1, sampling_rate * int(file_duration)))
        augmented_signal = transform(signal)
        filename = file_root + f'{gain}db.wav'
        audiofile.write(filename, augmented_signal, sampling_rate)
        gains.append(gain)
        files.append(filename)

    index = audformat.filewise_index(files)
    db.schemes["gain"] = audformat.Scheme("float")
    db["files"] = audformat.Table(index=index)
    db["files"]["gain"] = audformat.Column(scheme_id="gain")
    db["files"]["gain"].set(gains)
    db.save(db_root)
    return db


if __name__ == "__main__":
    print(create_white_noise_db('./test'))
