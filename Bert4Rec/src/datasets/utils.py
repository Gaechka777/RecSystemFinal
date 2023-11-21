import zipfile
import wget


def download(url, save_path):
    """

    Args:
        url: path to dataset for download
        save_path: path to save dataset

    """
    wget.download(url, str(save_path))


def unzip(zippath, save_path):
    """

    Args:
        zippath: path to dataset in zip format
        save_path: path to save dataset

    """
    zips = zipfile.ZipFile(zippath)
    zips.extractall(save_path)
    zips.close()


def get_count(tp, ids):
    """

    Args:
        tp: dataframe
        ids: name of feature to group

    Return:
        count of grouped features
    """
    groups = tp[[ids]].groupby(ids, as_index=False)
    count = groups.size()
    return count


