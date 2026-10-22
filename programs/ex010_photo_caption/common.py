"""ex010 の共通の処理（一覧画像の作成，処理速度の記録）．"""
import os

import numpy as np

from data import image_path


def plot_image_grid(ids: list, titles: list, suptitle: str, path: str, ncols: int = 4, size: int = 400,
                    wrap: int = 30) -> None:
    """画像を格子状に並べ，各画像に題を付けて保存する．

    Args:
        ids (list[int]): お題IDのリスト．
        titles (list[str]): 各画像の題（長い題は wrap 文字ごとに改行する）．
        suptitle (str): 図全体の題．
        path (str): 保存先のパス．
        ncols (int): 列数．
        size (int): 画像の一辺の画素数．
        wrap (int): 題の1行の文字数．

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from PIL import Image

    font = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"
    if os.path.exists(font):
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    nrows = int(np.ceil(len(ids) / ncols))
    n_lines = max(t.count("\n") + 1 + len(t) // wrap for t in titles)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, (3.6 + 0.16 * n_lines) * nrows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, fid, title in zip(np.ravel(axes), ids, titles):
        im = Image.open(image_path(fid)).convert("RGB")
        im.thumbnail((size, size))
        canvas = Image.new("RGB", (size, size), "white")
        canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
        ax.imshow(canvas)
        lines = []
        for part in title.split("\n"):
            lines += [part[i:i + wrap] for i in range(0, max(len(part), 1), wrap)]
        ax.set_title("\n".join(lines), fontsize=7, loc="left")
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def throughput(n_images: int, sec: float) -> dict:
    """処理速度を記録する辞書を作る．

    Args:
        n_images (int): 処理した枚数．
        sec (float): 秒数．

    Returns:
        dict: n_images，sec，images_per_sec，全写真（約337万枚）の処理時間の見積もり（時間）．
    """
    return {"n_images": n_images, "sec": sec, "images_per_sec": n_images / sec, "est_hours_all_photos": 3_371_140 / (n_images / sec) / 3600}
