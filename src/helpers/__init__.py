import requests


def web_healthcheck(url: str) -> bool:
    """Check if the given URL is reachable.

    Args:
        url (str): The URL to check.

    Returns:
        bool: True if the URL is reachable, False otherwise.
    """
    try:
        response = requests.head(url)
        return response.status_code == 200
    except requests.RequestException:
        return False
