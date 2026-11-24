"""Exact historical publications and the authority each release established."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
from importlib.resources import files

from .amendment_artifact import AmendmentArtifactError


class AmendmentRule(Enum):
    FOUNDING_ARTICLES = "genesis-required-for-every-amendment"
    RANKED_BOOKS = "ranked-books"
    SOVEREIGN_FRAME = "ranked-books-with-genesis-authorized-frame"
    PLATFORM_AUTHORED_BOOKS = "platform-authored-ranked-books"


@dataclass(frozen=True)
class V1Release:
    commit: str
    git_sha256: str
    crlf_sha256: str
    amendment_rule: AmendmentRule
    supports_activation: bool = False
    packaged_default: bool = False
    crlf: bool = False

    @property
    def digest(self) -> str:
        return self.crlf_sha256 if self.crlf else self.git_sha256

    @property
    def public_source_url(self) -> str:
        return (
            "https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/"
            f"{self.commit}/kestrel_sovereign/data/KESTREL_CONSTITUTION.md"
        )

    @property
    def resource_path(self) -> str:
        if self.packaged_default:
            return "data/KESTREL_CONSTITUTION.md"
        return f"data/constitutions/v1/{self.commit}.md"

    def authenticated_text(self) -> str:
        path = files("kestrel_sovereign").joinpath(*self.resource_path.split("/"))
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise AmendmentArtifactError(f"Cannot read the published edition-one source at {self.commit}") from exc
        # The original packaged default follows Git's checkout settings. An
        # exact CRLF checkout and the exact Git blob name the same publication,
        # but they remain separate governing byte strings and separate anchors.
        if self.packaged_default and hashlib.sha256(content).hexdigest() == self.crlf_sha256:
            content = content.replace(b"\r\n", b"\n")
        if hashlib.sha256(content).hexdigest() != self.git_sha256:
            raise AmendmentArtifactError(f"Installed edition-one source differs from its publication at {self.commit}")
        if self.crlf:
            content = content.replace(b"\n", b"\r\n")
            if hashlib.sha256(content).hexdigest() != self.crlf_sha256:
                raise AmendmentArtifactError(f"Cannot reproduce the recorded CRLF checkout at {self.commit}")
        return content.decode("utf-8")


V1_RELEASES = (
    V1Release(
        "34eacb73337ce4bcae4ce1d79242a4a4401f62ea",
        "54a30441d913c385a08f5d739261a9ac6fb4fa5f5d8f1f380b8e403504c7c254",
        "84adf6c65583d36c404eebe318a3785a77e29f54ede536ae79a9630346005d81",
        AmendmentRule.FOUNDING_ARTICLES,
    ),
    V1Release(
        "763e6ac2cf67c1643f6d5f8305f06da17a1e56c7",
        "90b1bcb03610b95a5cffacfc76b38fc4e9a3a3c66b0134596bdc0388683c016c",
        "0f9a8e83d363bb0d8ebc18b26b6c7f0d0c72e692c37cb08ab92e4a06879bdeb7",
        AmendmentRule.RANKED_BOOKS,
    ),
    V1Release(
        "0b24719bcec3368d97eff7829ada5051d4518b96",
        "5f364e0f9f98fcfc07f6c71c999dfe2d402e25ce12d7fc16964e65a13fc84f5a",
        "e4e9339d73603c7e829a723a79aabae483ea9e85ee3a1ba6920fb2d114cedd41",
        AmendmentRule.RANKED_BOOKS,
    ),
    V1Release(
        "84f4591abaab4e55a300bbe97864b1d925041884",
        "46687f754bc008f4022599fb481ebf9abf0fbf1e8e205472ac8efba95628eb9c",
        "7dd117bbf6416b97b0dde13336877bc010e2fa1225edee2b9bea4ee0998c30c3",
        AmendmentRule.RANKED_BOOKS,
        supports_activation=True,
    ),
    V1Release(
        "f0672b394ae2e202135d9a3181245f2806a56a8b",
        "08aec7530f912ad2ae154a3687387bd1a73b0489a3532d7f67c99d635372d63f",
        "58120eabb139f366a541c7859e1b262435ab8f066970bfc936ea5b366794f44b",
        AmendmentRule.SOVEREIGN_FRAME,
        supports_activation=True,
    ),
    V1Release(
        "a292b641893f102c6863f86a6ef0440ca8ae6cf0",
        "6948454b61253548b02b943dff166ba0c2f2eaabe2f542ace37424a8256d6467",
        "7408d91b0f00af15b178be219176cdd084e04cb0b8326f76bfa8ee41a126534f",
        AmendmentRule.PLATFORM_AUTHORED_BOOKS,
        supports_activation=True,
        packaged_default=True,
    ),
)
CURRENT_V1_RELEASE = V1_RELEASES[-1]
V1_CHECKOUTS = tuple(form for release in V1_RELEASES for form in (release, replace(release, crlf=True)))
PRE_DORMANCY_RELEASES = {release.digest: release for release in V1_CHECKOUTS if not release.supports_activation}
_RELEASE_BY_DIGEST = {release.digest: release for release in V1_CHECKOUTS}


def published_v1_release(digest: str) -> V1Release | None:
    return _RELEASE_BY_DIGEST.get(digest)
