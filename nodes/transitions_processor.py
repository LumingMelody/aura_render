"""
Transitions Processor Node

Specialized node for handling video transitions between shots and scenes.
This is included in the effects processor but kept as a separate node for modularity.
"""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .base_node import BaseNode, NodeConfig, NodeResult, ProcessingContext, NodeStatus
from .effects_processor import TransitionType, EffectsProcessorNode, EffectsConfig


@dataclass
class TransitionsConfig(NodeConfig):
    """Configuration for transitions processor node"""
    default_transition_duration: float = 0.5
    enable_smart_transitions: bool = True
    transition_quality: str = "high"  # low, medium, high, ultra


class TransitionsProcessorNode(EffectsProcessorNode):
    """Specialized transitions processor node"""
    
    def __init__(self, config: TransitionsConfig):
        # Convert to EffectsConfig for parent class
        effects_config = EffectsConfig(
            node_id=config.node_id,
            name=config.name,
            description=config.description,
            priority=config.priority,
            timeout=config.timeout,
            retry_count=config.retry_count,
            transition_duration=config.default_transition_duration,
            enable_transitions=True,
            quality_level=config.transition_quality
        )
        super().__init__(effects_config)
        self.transitions_config = config
        
    async def process(self, context: ProcessingContext) -> NodeResult:
        """Process only transitions (focused version of effects processing)"""
        try:
            self.logger.info("Starting transitions processing")
            
            # Extract input data
            shot_blocks = context.intermediate_results['shot_blocks']
            emotion_analysis = context.intermediate_results.get('emotion_analysis', {})
            
            # Focus only on transition generation
            effects_plan = await self._analyze_scene_requirements(
                shot_blocks, emotion_analysis, context
            )
            
            transitions = await self._design_scene_transitions(
                shot_blocks, effects_plan, context
            )
            
            # Prepare focused result for transitions only
            transitions_data = {
                'transitions': transitions,
                'transition_metadata': {
                    'total_transitions': len(transitions),
                    'average_duration': sum(t['duration'] for t in transitions) / len(transitions) if transitions else 0,
                    'transition_types_used': list(set(t['type'] for t in transitions)),
                    'smart_transitions_enabled': self.transitions_config.enable_smart_transitions
                }
            }
            
            return NodeResult(
                status=NodeStatus.COMPLETED,
                data=transitions_data,
                next_nodes=['effects_processor', 'render_compositor']
            )
            
        except Exception as e:
            self.logger.error(f"Transitions processing failed: {e}")
            return NodeResult(
                status=NodeStatus.FAILED,
                error_message=str(e)
            )