"""Safe Harbor identifier categories and the field schema that classifies a record.

HIPAA Safe Harbor (45 CFR 164.514(b)(2)) names eighteen identifier categories
that must be removed for a record to be de-identified, plus a condition the
pipeline cannot satisfy on its own: the operator must have no actual knowledge
that what remains could identify the individual. This module models the
eighteen categories and the per-field classification the pipeline needs to
know what each field of a record is. It deliberately has no notion of a
"healthcare mode": the categories are a generic assurance vocabulary that any
feature can require.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from kestrel_sovereign.deidentification.errors import DeidentificationConfigError


class SafeHarborIdentifier(Enum):
    """The eighteen identifier categories of 45 CFR 164.514(b)(2)(i)(A)-(R)."""

    NAMES = "names"  # (A)
    GEOGRAPHIC_SUBDIVISIONS = "geographic_subdivisions"  # (B)
    DATES = "dates"  # (C) date elements except year, and ages over 89
    TELEPHONE_NUMBERS = "telephone_numbers"  # (D)
    FAX_NUMBERS = "fax_numbers"  # (E)
    EMAIL_ADDRESSES = "email_addresses"  # (F)
    SOCIAL_SECURITY_NUMBERS = "social_security_numbers"  # (G)
    MEDICAL_RECORD_NUMBERS = "medical_record_numbers"  # (H)
    HEALTH_PLAN_BENEFICIARY_NUMBERS = "health_plan_beneficiary_numbers"  # (I)
    ACCOUNT_NUMBERS = "account_numbers"  # (J)
    CERTIFICATE_LICENSE_NUMBERS = "certificate_license_numbers"  # (K)
    VEHICLE_IDENTIFIERS = "vehicle_identifiers"  # (L)
    DEVICE_IDENTIFIERS = "device_identifiers"  # (M)
    WEB_URLS = "web_urls"  # (N)
    IP_ADDRESSES = "ip_addresses"  # (O)
    BIOMETRIC_IDENTIFIERS = "biometric_identifiers"  # (P)
    FULL_FACE_IMAGES = "full_face_images"  # (Q)
    OTHER_UNIQUE_IDENTIFIERS = "other_unique_identifiers"  # (R)


class FieldKind(Enum):
    """What a record field is, as declared by the caller's schema."""

    IDENTIFIER = "identifier"  # holds one Safe Harbor identifier category
    FREE_TEXT = "free_text"  # prose that may contain identifiers of any category
    NON_IDENTIFYING = "non_identifying"  # kept, but scanned for identifier patterns


class FieldRole(Enum):
    """How an identifier field is transformed.

    ``REMOVE`` drops the field. The other roles are the generalizations Safe
    Harbor permits instead of removal, and each is valid for one category only.
    """

    REMOVE = "remove"
    ZIP_CODE = "zip_code"  # GEOGRAPHIC: first three digits, or 000 when restricted
    DATE = "date"  # DATES: year only, aggregated when 90 or more years old
    BIRTH_DATE = "birth_date"  # DATES: as DATE; the role names the field's meaning
    AGE = "age"  # DATES: retained through 89, aggregated above


class TransformationAction(Enum):
    """What the pipeline did to one identifier occurrence."""

    REMOVED = "removed"  # the field was dropped from the output record
    GENERALIZED = "generalized"  # the value was coarsened (ZIP3, year, 90+)
    TRANSFORMED = "transformed"  # a free-text span was replaced by a placeholder


#: The single aggregate category Safe Harbor permits for ages over 89 and for
#: any date element (including a year) indicative of such an age.
AGE_90_OR_OLDER = "90+"

#: Three-digit ZIP prefixes whose combined geographic unit held 20,000 or fewer
#: people, which Safe Harbor requires to be reported as ``000``. This is the
#: list HHS published in its de-identification guidance, derived from the 2000
#: Census. It is configuration, not law: a pipeline built with a newer census
#: list records that list in its configuration digest.
HHS_RESTRICTED_ZIP3_PREFIXES = frozenset({
    "036", "059", "063", "102", "203", "556", "692", "790", "821",
    "823", "830", "831", "878", "879", "884", "890", "893",
})

_ROLE_CATEGORY = {
    FieldRole.ZIP_CODE: SafeHarborIdentifier.GEOGRAPHIC_SUBDIVISIONS,
    FieldRole.DATE: SafeHarborIdentifier.DATES,
    FieldRole.BIRTH_DATE: SafeHarborIdentifier.DATES,
    FieldRole.AGE: SafeHarborIdentifier.DATES,
}


@dataclass(frozen=True)
class FieldSpec:
    """The caller's classification of one record field.

    Every field of every record must be classified; the pipeline refuses a
    record carrying a field its schema does not name. Build specs with the
    constructors below rather than the raw dataclass.
    """

    kind: FieldKind
    identifier: Optional[SafeHarborIdentifier] = None
    role: FieldRole = FieldRole.REMOVE

    def __post_init__(self) -> None:
        if self.kind is FieldKind.IDENTIFIER:
            if self.identifier is None:
                raise DeidentificationConfigError(
                    "an identifier field must name its Safe Harbor category"
                )
            expected = _ROLE_CATEGORY.get(self.role)
            if expected is not None and expected is not self.identifier:
                raise DeidentificationConfigError(
                    f"field role {self.role.value!r} applies only to the "
                    f"{expected.value!r} category, not {self.identifier.value!r}"
                )
        elif self.identifier is not None or self.role is not FieldRole.REMOVE:
            raise DeidentificationConfigError(
                f"a {self.kind.value!r} field cannot carry an identifier "
                "category or transformation role"
            )

    @classmethod
    def remove(cls, identifier: SafeHarborIdentifier) -> "FieldSpec":
        """An identifier field dropped from the output entirely."""
        return cls(FieldKind.IDENTIFIER, identifier, FieldRole.REMOVE)

    @classmethod
    def zip_code(cls) -> "FieldSpec":
        """A ZIP code generalized to its permitted three-digit prefix."""
        return cls(
            FieldKind.IDENTIFIER,
            SafeHarborIdentifier.GEOGRAPHIC_SUBDIVISIONS,
            FieldRole.ZIP_CODE,
        )

    @classmethod
    def date(cls) -> "FieldSpec":
        """A date directly related to the individual, generalized to its year.

        A year 90 or more years before the run's reference date is aggregated
        to 90+: the person was alive then, so it implies an age over 89.
        """
        return cls(FieldKind.IDENTIFIER, SafeHarborIdentifier.DATES, FieldRole.DATE)

    @classmethod
    def birth_date(cls) -> "FieldSpec":
        """A birth date: its year, or the 90+ aggregate when the year implies it."""
        return cls(
            FieldKind.IDENTIFIER, SafeHarborIdentifier.DATES, FieldRole.BIRTH_DATE
        )

    @classmethod
    def age(cls) -> "FieldSpec":
        """An age in years: kept through 89, aggregated to 90+ above."""
        return cls(FieldKind.IDENTIFIER, SafeHarborIdentifier.DATES, FieldRole.AGE)

    @classmethod
    def free_text(cls) -> "FieldSpec":
        """Prose scrubbed span by span for every detectable category."""
        return cls(FieldKind.FREE_TEXT)

    @classmethod
    def non_identifying(cls) -> "FieldSpec":
        """A value the caller asserts identifies no one; refused if it does."""
        return cls(FieldKind.NON_IDENTIFYING)

    def to_dict(self) -> dict:
        """Canonical, content-free description recorded in the config digest."""
        return {
            "kind": self.kind.value,
            "identifier": self.identifier.value if self.identifier else None,
            "role": self.role.value,
        }
