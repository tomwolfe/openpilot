BASE_URL = "https://raw.githubusercontent.com/tomwolfe/ci-artifacts/0/"

def get_url(route_name: str, segment_num, filename: str) -> str:
  return BASE_URL + f"{route_name.replace('|', '/')}/{segment_num}/{filename}"
