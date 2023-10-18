import zipfile
import wget


def download(url, save_path):
    wget.download(url, str(save_path))


def unzip(zippath, save_path):
    zips = zipfile.ZipFile(zippath)
    zips.extractall(save_path)
    zips.close()


def get_count(tp, ids):
    groups = tp[[ids]].groupby(ids, as_index=False)
    count = groups.size()
    return count


