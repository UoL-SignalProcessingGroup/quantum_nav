

import requests

from pathlib import Path
from requests import HTTPError


def download_file(url: str, dest: Path):
    """
    Downloads a file from a URL and writes it to destination file.
    Commonly used by the toolbox for downloading database files from
    selected web-servers.

    :param url: The URL of the file to be downloaded.
    :type url: str

    :param dest: The file to save the downloaded file as.
    :type dest: Path
    """

    response = requests.get(url, allow_redirects=True)
    status_code = response.status_code

    if status_code == 200:
        with open(dest, 'wb') as f:
            f.write(response.content)
            return

    raise HTTPError(f'{url} responded with HTTP status code {status_code}')