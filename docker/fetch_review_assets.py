"""Install fixed browser assets inside the image, verifying the npm tarball."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile
from urllib.request import urlopen

VERSION = "0.180.0"
SOURCE = f"https://registry.npmjs.org/three/-/three-{VERSION}.tgz"
INTEGRITY = "sha512-o+qycAMZrh+TsE01GqWUxUIKR1AL0S8pq7zDkYOQw8GqfX8b8VoCKYUoHbhiX5j+7hr8XsuHDVU6+gkQJQKg9w=="
DESTINATION = Path("/opt/sensor-review-web")
FILES = {
    "package/build/three.module.js": "three.module.js",
    "package/build/three.core.js": "three.core.js",
    "package/examples/jsm/controls/OrbitControls.js": "OrbitControls.js",
    "package/LICENSE": "three-LICENSE.txt",
}
with urlopen(SOURCE, timeout=60) as response:
    archive = response.read()
actual = "sha512-" + base64.b64encode(hashlib.sha512(archive).digest()).decode()
if actual != INTEGRITY:
    raise RuntimeError("Three.js package integrity mismatch")
DESTINATION.mkdir(parents=True, exist_ok=True)
hashes = {}
with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as package:
    for member, name in FILES.items():
        content = package.extractfile(member).read()
        (DESTINATION / name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
(DESTINATION / "manifest.json").write_text(json.dumps({
    "package": "three", "version": VERSION, "source": SOURCE,
    "integrity": INTEGRITY, "files_sha256": hashes,
}, indent=2) + "\n")
print(f"Installed verified Three.js {VERSION} assets in {DESTINATION}")
