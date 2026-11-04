"""Download the official UCI Online Shoppers Purchasing Intention dataset."""

from pathlib import Path
from urllib.request import Request, urlopen

from purchase_intent.config import load_params


def main() -> None:
    """Download the source CSV declared in params.yaml."""
    params = load_params()
    destination = Path(params["data"]["raw_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(params["data"]["url"], headers={"User-Agent": "11MLET-Tech-Challenge/1.0"})
    with urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())
    print(f"Dataset baixado para {destination}")


if __name__ == "__main__":
    main()
