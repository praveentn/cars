# components/conscious_agent.py
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import json

from core.database import DatabaseManager, Event, AgentState, TaskPlan
from core.llm_client import llm_client

logger = logging.getLogger(__name__)

class ConsciousAgentComponent:
    """
    Conscious Agent (Deliberative Reasoner / System 2)
    - Handles explicit, slow, goal-directed reasoning
    - Breaks down problems into steps (chain-of-thought)
    - Generates explanations, evaluates trade-offs, and forms plans
    """
    
    def __init__(self):
        self.name = "conscious_agent"
        self.status = "idle"
        self.plans_created = 0
        self.plans_executed = 0
        self.reasoning_sessions = 0
        self.active_plans = {}
        self.tools_registry = {
            "retriever": self._call_retriever,
            "memory": self._call_memory,
            "self_agent": self._call_self_agent
        }
        
    async def start(self):
        """Start the conscious agent component"""
        self.status = "active"
        await self._update_state()
        logger.info("Conscious Agent component started")
        
        # Start background plan execution monitor
        asyncio.create_task(self._plan_monitor_loop())
    
    async def stop(self):
        """Stop the conscious agent component"""
        self.status = "stopped"
        await self._update_state()
        logger.info("Conscious Agent component stopped")
    
    async def plan(self, goal: str, context: Dict[str, Any] = None) -> str:
        """
        Create a task plan for achieving a goal
        
        Args:
            goal: The objective to achieve
            context: Additional context information
            
        Returns:
            Plan ID
        """
        try:
            plan_id = str(uuid.uuid4())
            
            # Generate plan using LLM
            plan_steps = await self._generate_plan(goal, context or {})
            
            # Create task plan
            task_plan = TaskPlan(
                id=plan_id,
                goal=goal,
                steps=plan_steps,
                status="created",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                agent_id=self.name
            )
            
            # Save to database (simulated with memory for now)
            self.active_plans[plan_id] = task_plan
            
            self.plans_created += 1
            await self._update_state()
            
            logger.info(f"Plan created: {plan_id} for goal: {goal}")
            return plan_id
            
        except Exception as e:
            logger.error(f"Error creating plan: {e}")
            raise
    
    async def execute(self, plan_id: str) -> Dict[str, Any]:
        """
        Execute a task plan
        
        Args:
            plan_id: ID of the plan to execute
            
        Returns:
            Execution results
        """
        try:
            if plan_id not in self.active_plans:
                raise ValueError(f"Plan {plan_id} not found")
            
            plan = self.active_plans[plan_id]
            plan.status = "executing"
            plan.updated_at = datetime.now()
            
            results = []
            
            # Execute each step
            for i, step in enumerate(plan.steps):
                step_result = await self._execute_step(step, plan_id, i)
                results.append(step_result)
                
                # Update step status
                step["status"] = "completed" if step_result["success"] else "failed"
                step["result"] = step_result
                
                # Stop execution if step failed and is critical
                if not step_result["success"] and step.get("critical", False):
                    plan.status = "failed"
                    break
            
            # Update plan status
            if plan.status != "failed":
                plan.status = "completed"
            
            plan.updated_at = datetime.now()
            self.plans_executed += 1
            await self._update_state()
            
            logger.info(f"Plan {plan_id} execution completed with status: {plan.status}")
            
            return {
                "plan_id": plan_id,
                "status": plan.status,
                "steps_completed": len([r for r in results if r["success"]]),
                "total_steps": len(plan.steps),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error executing plan {plan_id}: {e}")
            if plan_id in self.active_plans:
                self.active_plans[plan_id].status = "error"
            raise
    
    async def reason(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform explicit reasoning on input data
        
        Args:
            input_data: Data to reason about
            
        Returns:
            Reasoning results with explanations
        """
        try:
            self.reasoning_sessions += 1
            
            # Use ReAct (Reasoning + Acting) pattern
            reasoning_result = await self._react_reasoning(input_data)
            
            await self._update_state()
            
            logger.info(f"Reasoning session completed for input: {input_data}")
            
            return reasoning_result
            
        except Exception as e:
            logger.error(f"Error in reasoning: {e}")
            raise
    
    async def _generate_plan(self, goal: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate a step-by-step plan using LLM"""
        messages = [
            {
                "role": "system",
                "content": """You are a strategic planning expert. Create a detailed, actionable plan to achieve the given goal.
                
                Return a JSON array of steps, where each step has:
                {
                    "step_number": 1,
                    "action": "specific action to take",
                    "tool": "tool_name or null",
                    "parameters": {"key": "value"},
                    "expected_outcome": "what should happen",
                    "critical": true/false,
                    "estimated_time": "time estimate"
                }
                
                Available tools: retriever, memory, self_agent
                Make the plan specific and executable."""
            },
            {
                "role": "user",
                "content": f"Goal: {goal}\nContext: {json.dumps(context, indent=2)}"
            }
        ]
        
        try:
            response = await llm_client.chat_completion(messages, temperature=0.3)
            plan_data = json.loads(response["content"])
            
            # Validate and enhance plan steps
            enhanced_steps = []
            for step in plan_data:
                enhanced_step = {
                    "step_number": step.get("step_number", len(enhanced_steps) + 1),
                    "action": step.get("action", ""),
                    "tool": step.get("tool"),
                    "parameters": step.get("parameters", {}),
                    "expected_outcome": step.get("expected_outcome", ""),
                    "critical": step.get("critical", False),
                    "estimated_time": step.get("estimated_time", "unknown"),
                    "status": "pending",
                    "created_at": datetime.now().isoformat()
                }
                enhanced_steps.append(enhanced_step)
            
            return enhanced_steps
            
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return [
                {
                    "step_number": 1,
                    "action": f"Analyze and break down the goal: {goal}",
                    "tool": "self_agent",
                    "parameters": {"goal": goal, "context": context},
                    "expected_outcome": "Better understanding of requirements",
                    "critical": True,
                    "estimated_time": "2-5 minutes",
                    "status": "pending",
                    "created_at": datetime.now().isoformat()
                }
            ]
    
    async def _execute_step(self, step: Dict[str, Any], plan_id: str, step_index: int) -> Dict[str, Any]:
        """Execute a single plan step"""
        try:
            logger.info(f"Executing step {step_index + 1} of plan {plan_id}: {step['action']}")
            
            result = {
                "step_number": step["step_number"],
                "action": step["action"],
                "success": False,
                "output": None,
                "error": None,
                "execution_time": 0
            }
            
            start_time = datetime.now()
            
            # Execute tool if specified
            if step.get("tool") and step["tool"] in self.tools_registry:
                tool_func = self.tools_registry[step["tool"]]
                tool_result = await tool_func(step.get("parameters", {}))
                result["output"] = tool_result
                result["success"] = True
            else:
                # Simulate action execution
                await asyncio.sleep(0.5)  # Simulate processing time
                result["output"] = f"Executed: {step['action']}"
                result["success"] = True
            
            execution_time = (datetime.now() - start_time).total_seconds()
            result["execution_time"] = round(execution_time, 3)
            
            return result
            
        except Exception as e:
            return {
                "step_number": step["step_number"],
                "action": step["action"],
                "success": False,
                "output": None,
                "error": str(e),
                "execution_time": 0
            }
    
    async def _react_reasoning(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Implement ReAct (Reasoning + Acting) pattern"""
        reasoning_steps = []
        
        # Initial thought
        thought = await self._generate_thought(input_data, reasoning_steps)
        reasoning_steps.append({"type": "thought", "content": thought})
        
        # Action planning
        action_plan = await self._generate_action(thought, input_data)
        reasoning_steps.append({"type": "action", "content": action_plan})
        
        # Observation (simulated)
        observation = await self._simulate_observation(action_plan)
        reasoning_steps.append({"type": "observation", "content": observation})
        
        # Final reasoning
        conclusion = await self._generate_conclusion(reasoning_steps, input_data)
        reasoning_steps.append({"type": "conclusion", "content": conclusion})
        
        return {
            "input": input_data,
            "reasoning_steps": reasoning_steps,
            "conclusion": conclusion,
            "confidence": 0.8,  # Placeholder confidence score
            "timestamp": datetime.now().isoformat()
        }
    
    async def _generate_thought(self, input_data: Dict[str, Any], context: List[Dict]) -> str:
        """Generate initial thought about the input"""
        messages = [
            {
                "role": "system",
                "content": "Generate a thoughtful analysis of the given input. What are the key aspects to consider?"
            },
            {
                "role": "user",
                "content": f"Input: {json.dumps(input_data, indent=2)}"
            }
        ]
        
        response = await llm_client.chat_completion(messages, temperature=0.5)
        return response["content"]
    
    async def _generate_action(self, thought: str, input_data: Dict[str, Any]) -> str:
        """Generate action plan based on thought"""
        messages = [
            {
                "role": "system",
                "content": "Based on the thought, what specific actions should be taken? Be concrete and actionable."
            },
            {
                "role": "user",
                "content": f"Thought: {thought}\nInput: {json.dumps(input_data, indent=2)}"
            }
        ]
        
        response = await llm_client.chat_completion(messages, temperature=0.3)
        return response["content"]
    
    async def _simulate_observation(self, action_plan: str) -> str:
        """Simulate observation of action results"""
        return f"Observation: Action plan '{action_plan}' would likely result in improved understanding and structured approach to the problem."
    
    async def _generate_conclusion(self, reasoning_steps: List[Dict], input_data: Dict[str, Any]) -> str:
        """Generate final conclusion from reasoning process"""
        steps_summary = "\n".join([f"{step['type']}: {step['content']}" for step in reasoning_steps])
        
        messages = [
            {
                "role": "system",
                "content": "Based on the reasoning process, provide a clear, actionable conclusion."
            },
            {
                "role": "user",
                "content": f"Reasoning process:\n{steps_summary}\n\nOriginal input: {json.dumps(input_data, indent=2)}"
            }
        ]
        
        response = await llm_client.chat_completion(messages, temperature=0.2)
        return response["content"]
    
    # Tool interface methods
    async def _call_retriever(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call retriever tool"""
        return {"tool": "retriever", "result": "Retrieved relevant information", "parameters": parameters}
    
    async def _call_memory(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call memory tool"""
        return {"tool": "memory", "result": "Accessed memory systems", "parameters": parameters}
    
    async def _call_self_agent(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call self-agent tool"""
        return {"tool": "self_agent", "result": "Triggered self-agent exploration", "parameters": parameters}
    
    async def _plan_monitor_loop(self):
        """Background loop to monitor plan execution"""
        while self.status == "active":
            try:
                # Clean up completed plans older than 1 hour
                current_time = datetime.now()
                to_remove = []
                
                for plan_id, plan in self.active_plans.items():
                    age = (current_time - plan.updated_at).total_seconds() / 3600
                    if age > 1 and plan.status in ["completed", "failed", "cancelled"]:
                        to_remove.append(plan_id)
                
                for plan_id in to_remove:
                    del self.active_plans[plan_id]
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in plan monitor loop: {e}")
                await asyncio.sleep(60)
    
    async def _update_state(self):
        """Update component state in database"""
        state = AgentState(
            agent_name=self.name,
            status=self.status,
            last_activity=datetime.now(),
            metrics={
                "plans_created": float(self.plans_created),
                "plans_executed": float(self.plans_executed),
                "reasoning_sessions": float(self.reasoning_sessions),
                "active_plans": float(len(self.active_plans)),
                "success_rate": round(
                    (self.plans_executed / max(self.plans_created, 1)) * 100, 2
                )
            },
            config={
                "tools_available": list(self.tools_registry.keys())
            }
        )
        await DatabaseManager.update_agent_state(state)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current component status"""
        return {
            "name": self.name,
            "status": self.status,
            "plans_created": self.plans_created,
            "plans_executed": self.plans_executed,
            "reasoning_sessions": self.reasoning_sessions,
            "active_plans": len(self.active_plans),
            "tools_available": list(self.tools_registry.keys())
        }

