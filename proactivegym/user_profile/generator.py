"""
User Profile Generator for ProactiveGym.

Generates random user backgrounds with:
- Age (15-100)
- Country (random from world countries)
- Occupation (from predefined list)
- Personality (MBTI 16 types)
- Education Level (matched with age)
- LLM-generated detailed background description
"""

import random
import os
import openai
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..config import get_default_config


# World countries list
COUNTRIES = [
    "United States", "China", "Japan", "Germany", "United Kingdom",
    "France", "India", "Italy", "Brazil", "Canada",
    "Russia", "South Korea", "Australia", "Spain", "Mexico",
    "Indonesia", "Netherlands", "Saudi Arabia", "Turkey", "Switzerland",
    "Poland", "Sweden", "Belgium", "Argentina", "Norway",
    "Austria", "United Arab Emirates", "Israel", "Ireland", "Singapore",
    "Malaysia", "South Africa", "Philippines", "Denmark", "Colombia",
    "Egypt", "Pakistan", "Chile", "Finland", "Vietnam",
    "Czech Republic", "Portugal", "New Zealand", "Greece", "Peru",
    "Romania", "Ukraine", "Hungary", "Thailand", "Nigeria",
]

# MBTI personality descriptions
MBTI_DESCRIPTIONS = {
    "ISTJ": "responsible and detail-oriented",
    "ISFJ": "supportive and reliable",
    "INFJ": "insightful and principled",
    "INTJ": "strategic and independent",
    "ISTP": "practical and observant",
    "ISFP": "gentle and sensitive",
    "INFP": "idealistic and empathetic",
    "INTP": "analytical and objective",
    "ESTP": "energetic and pragmatic",
    "ESFP": "spontaneous and enthusiastic",
    "ENFP": "creative and sociable",
    "ENTP": "inventive and outspoken",
    "ESTJ": "organized and logical",
    "ESFJ": "caring and sociable",
    "ENFJ": "charismatic and empathetic",
    "ENTJ": "decisive and ambitious",
}


@dataclass
class UserProfile:
    """User profile data."""
    age: int
    country: str
    occupation: str
    personality: str
    education_level: str
    gender: str
    background_description: str


class UserProfileGenerator:
    """Generates random user profiles."""

    def __init__(
        self,
        occupations_file: Optional[str] = None,
        personalities_file: Optional[str] = None,
        education_file: Optional[str] = None,
        use_llm: bool = True
    ):
        """
        Initialize the generator.

        Args:
            occupations_file: Path to occupations.txt file
            personalities_file: Path to personalities.txt file
            education_file: Path to education_levels.txt file
            use_llm: Whether to use LLM for detailed descriptions
        """
        base_dir = os.path.dirname(__file__)

        if occupations_file is None:
            occupations_file = os.path.join(base_dir, "occupations.txt")
        if personalities_file is None:
            personalities_file = os.path.join(base_dir, "personalities.txt")
        if education_file is None:
            education_file = os.path.join(base_dir, "education_levels.txt")

        self.occupations = self._load_list_file(occupations_file)
        self.personalities = self._load_list_file(personalities_file)
        self.education_levels = self._load_list_file(education_file)
        self.countries = COUNTRIES
        self.use_llm = use_llm

        # Load config for LLM
        if use_llm:
            self.config = get_default_config()
            self.client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url
            )

    def _load_list_file(self, filepath: str) -> List[str]:
        """Load list from file."""
        with open(filepath, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def generate(self, seed: Optional[int] = None) -> UserProfile:
        """
        Generate a random user profile.

        Args:
            seed: Random seed for reproducibility

        Returns:
            UserProfile with all attributes
        """
        if seed is not None:
            random.seed(seed)

        age = random.randint(15, 100)
        country = random.choice(self.countries)
        occupation = random.choice(self.occupations)
        personality = random.choice(self.personalities)
        gender = random.choice(["male", "female"])

        # Education level based on age
        if 15 <= age <= 18:
            education_level = "High School"
        else:
            education_level = random.choice(self.education_levels)

        # If age is 15-18 and occupation is a student type, must be High School Student
        if 15 <= age <= 18:
            if occupation in ["University Student", "Graduate Student", "High School Student"]:
                occupation = "High School Student"

        background_description = self._generate_description(
            age, country, occupation, personality, education_level, gender
        )

        return UserProfile(
            age=age,
            country=country,
            occupation=occupation,
            personality=personality,
            education_level=education_level,
            gender=gender,
            background_description=background_description
        )

    def _generate_description(
        self,
        age: int,
        country: str,
        occupation: str,
        personality: str,
        education_level: str,
        gender: str
    ) -> str:
        """Generate a natural language background description using LLM."""
        if self.use_llm:
            return self._generate_description_llm(age, country, occupation, personality, education_level, gender)
        else:
            return self._generate_description_template(age, country, occupation, personality, education_level, gender)

    def _generate_description_llm(
        self,
        age: int,
        country: str,
        occupation: str,
        personality: str,
        education_level: str,
        gender: str
    ) -> str:
        """Generate detailed description using LLM."""
        personality_desc = MBTI_DESCRIPTIONS.get(personality, "unique")
        pronoun = "he" if gender == "male" else "she"
        pronoun_pos = "his" if gender == "male" else "her"

        prompt = f"""Based on the following user profile, generate a detailed and vivid background description in 3-4 sentences. Make it feel like a real person with a unique story.

User Profile:
- Gender: {gender}
- Age: {age}
- Country: {country}
- Occupation: {occupation}
- Education: {education_level}
- Personality traits: {personality_desc}

Requirements:
1. DO NOT mention the personality type code (like INTJ, ENFP, etc.)
2. Instead, naturally describe their personality through their behavior and characteristics
3. Include plausible past experiences or background story that shaped who they are
4. Mention their current life situation, habits, or daily routine
5. Keep it concise but vivid (3-4 sentences)
6. Write in third person, use "{pronoun}/{pronoun_pos}" as pronouns (NOT "they/their")
7. Start with "This user is a {age}-year-old {gender}..."

Generate the description:"""

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_completion_tokens=500,
                timeout=30
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[UserProfileGenerator] LLM error: {e}, falling back to template")
            return self._generate_description_template(age, country, occupation, personality, education_level, gender)

    def _generate_description_template(
        self,
        age: int,
        country: str,
        occupation: str,
        personality: str,
        education_level: str,
        gender: str
    ) -> str:
        """Generate description using template (fallback)."""
        pronoun = "He" if gender == "male" else "She"
        pronoun_pos = "his" if gender == "male" else "her"

        # Age group description
        if age < 18:
            age_desc = f"a {age}-year-old {gender} teenager"
        elif age < 25:
            age_desc = f"a {age}-year-old {gender} young adult"
        elif age < 35:
            age_desc = f"a {age}-year-old {gender} in early career"
        elif age < 50:
            age_desc = f"a {age}-year-old {gender} experienced professional"
        elif age < 65:
            age_desc = f"a {age}-year-old {gender} senior professional"
        else:
            age_desc = f"a {age}-year-old {gender}"

        # Personality description
        personality_desc = MBTI_DESCRIPTIONS.get(personality, "unique")

        # Build description
        if occupation in ["Retired", "Unemployed"]:
            if occupation == "Retired":
                occupation_part = "currently retired"
            else:
                occupation_part = "currently unemployed and seeking opportunities"
        elif occupation in ["University Student", "Graduate Student", "High School Student"]:
            occupation_part = f"currently a {occupation.lower()}"
        else:
            article = "an" if occupation[0].lower() in "aeiou" else "a"
            occupation_part = f"working as {article} {occupation.lower()}"

        desc = (
            f"This user is {age_desc} from {country}, {occupation_part}. "
            f"{pronoun} has a {education_level} education and {pronoun_pos} personality is {personality_desc}."
        )

        return desc

    def generate_batch(self, n: int, seed: Optional[int] = None) -> List[UserProfile]:
        """
        Generate multiple user profiles.

        Args:
            n: Number of profiles to generate
            seed: Random seed

        Returns:
            List of UserProfile objects
        """
        if seed is not None:
            random.seed(seed)

        return [self.generate() for _ in range(n)]


def generate_user_background(seed: Optional[int] = None, use_llm: bool = True) -> Dict[str, Any]:
    """
    Convenience function to generate a user background.

    Args:
        seed: Random seed
        use_llm: Whether to use LLM for detailed descriptions

    Returns:
        Dictionary with user profile data
    """
    generator = UserProfileGenerator(use_llm=use_llm)
    profile = generator.generate(seed)

    return {
        "age": profile.age,
        "gender": profile.gender,
        "country": profile.country,
        "occupation": profile.occupation,
        "personality": profile.personality,
        "education_level": profile.education_level,
        "background": profile.background_description
    }


def generate_and_save_csv(
    n: int,
    output_path: Optional[str] = None,
    use_llm: bool = True,
    seed: Optional[int] = None
) -> str:
    """
    Generate multiple user profiles and save to CSV.

    Args:
        n: Number of profiles to generate
        output_path: Path to save CSV file. If None, saves to data folder.
        use_llm: Whether to use LLM for detailed descriptions
        seed: Random seed

    Returns:
        Path to saved CSV file
    """
    import csv
    from datetime import datetime

    if output_path is None:
        # Save to data folder
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        os.makedirs(data_dir, exist_ok=True)
        filename = f"user_profiles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_path = os.path.join(data_dir, filename)

    generator = UserProfileGenerator(use_llm=use_llm)

    if seed is not None:
        random.seed(seed)

    profiles = []
    for i in range(n):
        print(f"Generating profile {i+1}/{n}...")
        profile = generator.generate()
        profiles.append(profile)

    # Write to CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(["id", "age", "gender", "country", "occupation", "personality", "education_level", "background_description"])
        # Data
        for i, profile in enumerate(profiles):
            writer.writerow([
                i + 1,
                profile.age,
                profile.gender,
                profile.country,
                profile.occupation,
                profile.personality,
                profile.education_level,
                profile.background_description
            ])

    print(f"\nSaved {n} profiles to: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate user profiles")
    parser.add_argument("-n", type=int, default=3, help="Number of profiles to generate")
    parser.add_argument("--save", action="store_true", help="Save to CSV file")
    parser.add_argument("--no-llm", action="store_true", help="Don't use LLM for descriptions")
    args = parser.parse_args()

    use_llm = not args.no_llm

    if args.save:
        # Generate and save to CSV
        generate_and_save_csv(n=args.n, use_llm=use_llm)
    else:
        # Just print profiles
        print("=" * 70)
        print("User Profile Generator Test")
        print("=" * 70)

        generator = UserProfileGenerator(use_llm=use_llm)

        for i in range(args.n):
            profile = generator.generate()
            print(f"\nProfile {i+1}:")
            print(f"  Age: {profile.age}")
            print(f"  Gender: {profile.gender}")
            print(f"  Country: {profile.country}")
            print(f"  Occupation: {profile.occupation}")
            print(f"  Personality: {profile.personality}")
            print(f"  Education: {profile.education_level}")
            print(f"\n  Description:\n  {profile.background_description}")
            print("-" * 70)
