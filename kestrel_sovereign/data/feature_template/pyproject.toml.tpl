[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "kestrel-feature-{{name_dashed}}"
version = "0.1.0"
description = "Kestrel feature: {{name}}"
requires-python = ">=3.11"
dependencies = ["kestrel-sovereign"]

[project.optional-dependencies]
test = ["pytest", "pytest-asyncio"]

[project.entry-points."kestrel_sovereign.features"]
{{class_name}} = "{{pkg_name}}.feature:{{class_name}}"

[tool.setuptools.packages.find]
where = ["src"]
