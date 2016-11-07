# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Utilities for the Web UI

TODO:
* Add a function to determine whether the two IPs are in the same sub-network.
  References:
  + Stackoverflow: How can I check if an IP is in a network in Python?
    https://stackoverflow.com/q/819355/4856091
"""


from urllib.parse import urlparse
import socket


def get_host_ip(url):
    """
    This function parses the input URL to get the hostname (or an IP),
    then the hostname is further resolved to its IP address.

    Parameters
    ----------
    url : str
        An URL string, which generally has the following format:
        ``scheme://netloc/path;parameters?query#fragment``
        while the ``netloc`` may look like ``user:pass@example.com:port``

    Returns
    -------
    ip : str
        An IPv4 address string.
        If something wrong happens (e.g., ``gaierror``), then ``None``
        is returned.
    """
    netloc = urlparse(url).netloc
    hostname = netloc.split("@")[-1].split(":")[0]
    try:
        ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        ip = None
    return ip


def get_local_ip(host="localhost", timeout=3.0):
    """
    Get the local IP address of this machine where this script runs.

    A dummy socket will be created and connects to the given host, then
    the valid local IP address used in this connection can be obtained.

    Parameters
    ----------
    host : str
        The host to which will be connected by a dummy socket, in order
        to determine the valid IP address.
    timeout : float
        Timeout (in seconds) on the blocking socket operations (e.g.,
        ``connect()``)

    Returns
    -------
    ip : str
        The local IPv4 address of this machine as a string.
        If something wrong happens (e.g., ``gaierror``), then ``None``
        is returned.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.settimeout(timeout)
            # Use 0 as the port will let OS determine the free port
            s.connect((host, 0))
            ip = s.getsockname()[0]
        except (socket.gaierror, socket.timeout):
            ip = None
    return ip