"""Artistic Figure Study Layer - identity-preserving professional
prompt composition, concept stacking and workflow payload builders for
character, garment-rendering, skin-texture, pose and form/proportion
study work.  Strictly additive: nothing here changes existing flows."""

from .batch_render import render_batch
from .figure_study import (
                              PRODUCTION_PASSES,
                              VARIATION_AXES,
                              axes_for_variation_strength,
                              list_variation_axes,
                              plan_figure_study_batch,
                              plan_production_batch,
                              resolve_passes,
                              write_batch_manifest,
                              write_production_batch,
)
from .prompt_composer import (
                              NEGATIVE_CONTROL_CHAINS,
                              RECIPES,
                              ArtisticLoraLayer,
                              ArtisticPromptRecipe,
                              ComposedArtisticPrompt,
                              compose_artistic_prompt,
                              compose_negative_chains,
                              get_recipe,
                              list_negative_chains,
                              list_recipes,
)

__all__ = [
    "ArtisticLoraLayer", "ArtisticPromptRecipe", "ComposedArtisticPrompt",
    "NEGATIVE_CONTROL_CHAINS", "PRODUCTION_PASSES", "RECIPES",
    "VARIATION_AXES", "axes_for_variation_strength",
    "compose_artistic_prompt", "compose_negative_chains", "get_recipe",
    "list_negative_chains", "list_recipes", "list_variation_axes",
    "plan_figure_study_batch", "plan_production_batch", "resolve_passes",
    "write_batch_manifest", "write_production_batch", "render_batch",
]
