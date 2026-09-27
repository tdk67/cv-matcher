"""Synthetic CV persona generator.

Generates deterministic, clearly-fictional candidate personas for testing
the RAG pipeline. All data is invented by a seeded PRNG - no real personal
data is used anywhere. Persona names are composed from curated first/last
name lists, so even across runs the output contains no real person's data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# Curated name/skill/role pools. All entries are generic, invented, or
# public-domain-style names - no real individuals.
_FIRST_NAMES = [
    "Alex", "Jamie", "Jordan", "Taylor", "Morgan", "Casey", "Riley",
    "Avery", "Quinn", "Dana", "Rowan", "Sasha", "Noor", "Mina",
    "Luka", "Ivy", "Ezra", "Mara", "Theo", "Nia",
]

_LAST_NAMES = [
    "Chen", "Smith", "Garcia", "Ito", "Kowalski", "Mohammed", "Novak",
    "Okafor", "Petrov", "Rossi", "Santos", "Tanaka", "Vargas", "Weber",
    "Yamamoto", "Dubois", "Andersen", "Haddad", "Lopez", "Silva",
]

_ROLES = [
    "Backend Engineer", "Frontend Engineer", "Full Stack Developer",
    "Data Scientist", "ML Engineer", "DevOps Engineer", "QA Engineer",
    "Product Manager", "Tech Lead", "Data Engineer", "Mobile Developer",
    "Security Engineer", "Cloud Architect", "Analytics Engineer",
    "Platform Engineer", "Site Reliability Engineer", "ML Ops Engineer",
    "Software Architect",
]

_SKILLS = [
    "Python", "Java", "Kotlin", "Go", "Rust", "TypeScript", "JavaScript",
    "React", "Angular", "Spring Boot", "Django", "FastAPI", "PostgreSQL",
    "MongoDB", "Redis", "Kubernetes", "Docker", "Terraform", "AWS", "GCP",
    "Azure", "Kafka", "RabbitMQ", "TensorFlow", "PyTorch", "scikit-learn",
    "Pandas", "NumPy", "Spark", "Airflow", "dbt", "CI/CD", "Jenkins",
    "GitHub Actions", "Prometheus", "Grafana", "OpenTelemetry", "gRPC",
    "GraphQL", "REST", "Linux", "Bash", "Ansible", "SQL", "NoSQL",
    "Elasticsearch", "ChromaDB", "LangChain", "LLM", "RAG",
]

_COMPANIES = [
    "Acme Robotics", "Northwind Digital", "Globex Systems", "Initech Labs",
    "Stark Analytics", "Umbrella Data", "Wayne Enterprises IT", "Hooli Cloud",
    "Pied Piper Tech", "Vandelay Industries", "Sirius Solutions",
    "Meridian Software", "Orbit Computing", "Vertex AI Group", "Cygnus Cloud",
    "Atlas Fintech",
]

_DEGREES = [
    "B.Sc. Computer Science", "B.Eng. Software Engineering",
    "M.Sc. Data Science", "M.Sc. Artificial Intelligence",
    "B.Sc. Information Systems", "M.Eng. Distributed Systems",
    "MBA Technology Management", "Ph.D. Machine Learning",
]

_CERTIFICATIONS = [
    "AWS Certified Solutions Architect", "Google Cloud Professional Engineer",
    "Certified Kubernetes Administrator", "CKA", "CKS",
    "Certified Scrum Master", "PMP", "Oracle Certified Java Developer",
    "Microsoft Azure DevOps Expert", "TensorFlow Developer Certificate",
]


@dataclass
class Experience:
    role: str
    company: str
    years: int


@dataclass
class Persona:
    """A single synthetic candidate persona."""

    name: str
    title: str
    email: str
    phone: str
    location: str
    summary: str
    skills: list[str] = field(default_factory=list)
    experience: list[Experience] = field(default_factory=list)
    education: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)

    # --- Serialization helpers -------------------------------------------
    def to_text(self) -> str:
        """Render the persona as a plain-text CV (markdown-flavored)."""
        lines = [
            f"# {self.name}",
            f"**{self.title}**",
            f"{self.location} | {self.email} | {self.phone}",
            "",
            "## Summary",
            self.summary,
            "",
            "## Skills",
            *[f"- {s}" for s in self.skills],
            "",
            "## Experience",
        ]
        for exp in self.experience:
            lines.append(f"- {exp.role} at {exp.company} ({exp.years} years)")
        lines.append("")
        lines.append("## Education")
        lines.extend(f"- {d}" for d in self.education)
        if self.certifications:
            lines.append("")
            lines.append("## Certifications")
            lines.extend(f"- {c}" for c in self.certifications)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Row-friendly mapping for CSV export."""
        return {
            "name": self.name,
            "title": self.title,
            "email": self.email,
            "location": self.location,
            "skills": "; ".join(self.skills),
            "summary": self.summary,
            "companies": "; ".join(e.company for e in self.experience),
            "years_experience": sum(e.years for e in self.experience),
        }


_LOCATIONS = [
    "Berlin, Germany", "Amsterdam, Netherlands", "Lisbon, Portugal",
    "Warsaw, Poland", "Madrid, Spain", "Prague, Czechia",
    "Bucharest, Romania", "Athens, Greece", "Tallinn, Estonia",
    "Vienna, Austria", "Zurich, Switzerland", "Stockholm, Sweden",
]

_DOMAIN_HINTS = {
    "Data Scientist": ["machine learning", "statistics", "ETL", "modeling"],
    "ML Engineer": ["ML pipelines", "model serving", "training"],
    "Backend Engineer": ["distributed systems", "APIs", "microservices"],
    "Frontend Engineer": ["UI", "React", "accessibility"],
    "Full Stack Developer": ["full-stack", "web platform", "product"],
    "DevOps Engineer": ["CI/CD", "infrastructure", "automation"],
    "QA Engineer": ["test automation", "quality", "regression"],
    "Product Manager": ["roadmap", "stakeholders", "product discovery"],
    "Tech Lead": ["architecture", "mentoring", "technical direction"],
    "Data Engineer": ["pipelines", "warehousing", "data infrastructure"],
    "Mobile Developer": ["Android", "iOS", "mobile apps"],
    "Security Engineer": ["security review", "threat modeling", "pentesting"],
    "Cloud Architect": ["cloud migration", "cost optimization", "architecture"],
    "Analytics Engineer": ["dbt", "analytics", "data products"],
    "Platform Engineer": ["developer experience", "internal platforms"],
    "Site Reliability Engineer": ["reliability", "SLOs", "on-call"],
    "ML Ops Engineer": ["model deployment", "feature stores", "monitoring"],
    "Software Architect": ["system design", "tech strategy", "architecture"],
}


def _persona(rng: random.Random, idx: int, max_experience: int = 4) -> Persona:
    """Build one random-but-seeded persona."""
    name = f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"
    title = rng.choice(_ROLES)
    domain = _DOMAIN_HINTS.get(title, ["software", "engineering"])
    email = f"{name.lower().replace(' ', '.')}.{idx}@example.org"
    phone = f"+{rng.randint(10, 49)} {rng.randint(100, 999)} {rng.randint(100, 999)} {rng.randint(100, 999)}"
    location = rng.choice(_LOCATIONS)

    skills = rng.sample(_SKILLS, k=rng.randint(8, 14))
    num_exp = rng.randint(1, max_experience)
    experience = []
    for _ in range(num_exp):
        experience.append(
            Experience(
                role=rng.choice(_ROLES),
                company=rng.choice(_COMPANIES),
                years=rng.randint(1, 5),
            )
        )
    education = rng.sample(_DEGREES, k=rng.randint(1, 2))
    certifications = rng.sample(_CERTIFICATIONS, k=rng.randint(0, 3))

    summary = (
        f"{title} with {sum(e.years for e in experience) or 2}+ years of "
        f"experience across {'/'.join(e.company for e in experience) or 'multiple product teams'}. "
        f"Focused on {', '.join(rng.sample(domain, k=min(2, len(domain))))}, "
        f"skilled in {', '.join(skills[:4])}."
    )

    return Persona(
        name=name,
        title=title,
        email=email,
        phone=phone,
        location=location,
        summary=summary,
        skills=skills,
        experience=experience,
        education=education,
        certifications=certifications,
    )


def generate_personas(count: int, seed: int = 0) -> list[Persona]:
    """Generate `count` deterministic personas (same seed -> same people)."""
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    rng = random.Random(seed)
    return [_persona(rng, i) for i in range(count)]