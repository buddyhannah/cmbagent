
import os
import pandas as pd

from IPython.display import Image
from autogen.agentchat.group import ContextVariables
from autogen.agentchat.group.patterns import AutoPattern

from typing import Dict, List, Literal
from autogen import Agent
from autogen.agentchat.group import ReplyResult, AgentTarget, Handoffs, TerminateTarget
from autogen import register_function




def register_nuclear_functions(namac_instance):
    
    """Register nuclear-specific functions with the updater agent"""
    diagnosis_agent = namac_instance.agents['diagnosis']
    updater_agent = namac_instance.agents['updater']
    
    if namac_instance.is_planning_and_control:
        control_agent = namac_instance.agents.get('control')
        planner_agent = namac_instance.agents.get('planner')
        admin_agent = namac_instance.agents.get('admin')

        #summarizer_agent = namac_instance.agents['summarizer']
    

    def extract_plan_from_planner(plan: str, context_variables: ContextVariables) -> ReplyResult:
        """
        Store planner's plan in control's context.
        
        Args:
            plan (str): The plan to store
            context_variables (ContextVariables): The context variables
        
        Returns:
            ReplyResult: Transfer to control agent with the plan in context
        """
     
        # Get plan
          
        context_variables.update({
            "final_plan": plan,
            
            "current_step_number": 0,
            "current_step_status": "complete",
            "current_sub_task": "Create a plan to answer the user's query.",
            "current_agent": "planner"
        })
        
        
        # Plan empty.Transfer to admin
        if plan.strip() == "":
            return ReplyResult(
                target=AgentTarget(admin_agent), 
                message="Planner failed to generate a valid plan. Transferring to admin for review.",
                context_variables=context_variables
                
            )
    
        # Plan good. Transfer to control
        return ReplyResult(
            target=AgentTarget(control_agent),
            message=f"{plan}",
            context_variables=context_variables
        )
        
        
       
    def record_status(
            current_step_status: Literal["complete", "failed"],
            current_step_number: int,
            next_task: str,
            next_agent: Literal["diagnosis", "strategy_inventory", "prognosis", 
                                    "strategy_assessment", "admin", "updater_helper", "planner","summarizer"],
            context_variables: ContextVariables,
         
        ) -> ReplyResult:
            """
            Updates execution context and manages agent transfers for nuclear reactor control.
            """
            
            
            # Map statuses to icons
            status_icons = {
                "failed": "❌",
                "complete": "✅",
                "in progress": "⏳"
            }
            last_icon = status_icons.get(current_step_status, "")
            
                
            # Update the context variables
            context_variables.update({
                "current_step_number": current_step_number + 1,
                "current_step_status": "in progress",
                "current_sub_task": next_task,
                "current_agent": next_agent,
            })
            
            
            # Step complete: Transfer to next agent
            if current_step_status == "complete":
                
                
                return ReplyResult(
                    target=AgentTarget(namac_instance.agents[next_agent].agent),
                    message=f"""
                        **Last Step #{current_step_number}:** {current_step_status} {last_icon}
                        **Current Step #{current_step_number + 1}:** in progress ⏳
                        - Current Task: {next_task}
                        - Current Agent: `{next_agent}`
                    """,
                    context_variables=context_variables
                )
            
            
            # Step failed: Transfer to admin
            return ReplyResult(
                    target=AgentTarget(namac_instance.agents[next_agent].agent),
                    message=f"""
                        Step failed. Transfer to admin.
                    """,
                    context_variables=context_variables
            )
           
            
            
            '''
            # Print context snapshot
            print("\n[DEBUG] Current Context Variables:")
            for k, v in context_variables.items():
                print(f"  {k}: {v}")
           
            # Print agent message history

            print(f"\n[DEBUG] Current agent message history for {next_agent}:")
            agent = namac_instance.agents[next_agent]
            for key in list(agent.agent.chat_messages.keys()):
                if agent.agent.chat_messages[key]:
                    print("messages before filtering")
                    for i, msg in enumerate(agent.agent.chat_messages[key]):
                        role = msg.get('role', 'unknown')
                        content_preview = str(msg.get('content', 'no-content'))[:100] + "..." if len(str(msg.get('content', ''))) > 100 else msg.get('content', 'no-content')
                        tool_calls = f", tool_calls: {len(msg.get('tool_calls', []))}" if msg.get('tool_calls') else ""
                        print(f"  {i}: [{role}] {content_preview}{tool_calls}")
            '''    


    def update_reactor_state(action: str) -> ReplyResult:
        """Apply an action to the reactor state, save it, and hand off to diagnosis agent."""
        
        df = pd.read_csv(namac_instance.reactor_state_file)
        current = df.iloc[-1].to_dict()

        print(f"[Updater] Applying action: {action}")

        actions = {
            "add_coolant": lambda s: {
                **s,
                'coolant_level': min(100, s['coolant_level'] + 15),
                'temperature': max(0, s['temperature'] - 5),
                'pressure': min(100, s['pressure'] + 5)
            },
            "reduce_power": lambda s: {
                **s,
                'power_output': max(0, s['power_output'] - 15),
                'temperature': max(0, s['temperature'] - 5),
                'coolant_level': max(0, s['coolant_level'] - 5),
                'pressure': max(0, s['pressure'] - 5)
            },
            "vent_pressure": lambda s: {
                **s,
                'pressure': max(0, s['pressure'] - 15),
                'temperature': min(100, s['temperature'] + 5),
                'coolant_level': max(0, s['coolant_level'] - 5)
            },
            "scram": lambda s: {
                **s,
                'power_output': 0,
                'temperature': max(0, s['temperature'] - 20),
                'moderator_level': max(0, s['moderator_level'] - 10),
                'coolant_level': max(0, s['coolant_level'] - 5),
                'pressure': max(0, s['pressure'] - 5),
            },
            "increase_power": lambda s: {
                **s,
                'power_output': min(100, s['power_output'] + 10),
                'temperature': min(100, s['temperature'] + 5),
                'pressure': min(100, s['pressure'] + 10),
                'moderator_level': max(0, s['coolant_level'] - 5),
            },
        }
        
        # Apply action
        new_state = actions.get(action, lambda s: s)(current)

        # Append updated state to CSV
        pd.DataFrame([new_state]).to_csv(
            namac_instance.reactor_state_file,
            mode='a',
            header=not os.path.exists(namac_instance.reactor_state_file),
            index=False
        )

        
        # TODO Clear history of non-tool agents
        '''
        for myName in ['strategy_assessment','prognosis', 'strategy_inventory']:
            print(f"cleared {myName}")
            namac_instance.agents[myName].agent.clear_history()
        
        '''
        
        # planning and control mode, hand off to control
        if namac_instance.is_planning_and_control:
            return ReplyResult(
                target=AgentTarget(namac_instance.agents['control'].agent),
                message=f"Action '{action}' applied successfully", data={"new_state": new_state} 
            )
            
        
        
        # Hardcoded flow mode: Hand off to diagnosis
        updated_context = ContextVariables(data={
            'main_task':"Get the nuclear reactor state and identify the SSFs",
            "work_dir": namac_instance.work_dir,
        })
        
        return ReplyResult(
            target=AgentTarget(namac_instance.agents['diagnosis'].agent),
            message=f"Action '{action}' applied. Reactor state updated. Passing to diagnosis.",
            context_variables=updated_context
        )
        
        
    def read_reactor_state(context_variables: ContextVariables):
        try:
            df = pd.read_csv(namac_instance.reactor_state_file)
            state = df.iloc[-1].to_dict()
            
            
            # Identify safety issues
            ssfs = []
            if state["temperature"] > 85:
                ssfs.append("High Temperature")
            if state["pressure"] > 90:
                ssfs.append("High Pressure")
            if state["coolant_level"] < 15:
                ssfs.append("Low Coolant")
            if state["power_output"] > 80:
                ssfs.append("High Power Output")
            if state["moderator_level"] < 20:
                ssfs.append("Low Neutron Moderator")

            ssfs = ssfs if ssfs else ["None"]
            

            # Print state for user
            state_list = list(state.items())
            state_bullet = [f"{key}: {value}" for key, value in state_list]
            print(
                    "*** Reactor State ***\n"
                    + "\n".join(f"- {v}" for v in state_bullet)
                    + "\n\n*** Safety Significant Factors ***\n"
                    + "\n".join(f"- {f}" for f in ssfs)
                )
            
            
            # Update context 
            context_variables.update({
                "reactor_state": state,
                "ssfs": ssfs,
                "current_step_status": "complete"
            })
        
            message = "Reactor state and SSFs have been updated."
            if not namac_instance.is_planning_and_control:
                message += " Please provide available strategies."
                
                
            # Transfer to next agent
            next_agent = control_agent if namac_instance.is_planning_and_control else namac_instance.agents.get('strategy_inventory')
            return ReplyResult(
                target=AgentTarget(next_agent),
                message=message,
                context_variables=context_variables
            )
            
        except Exception as e:
            context_variables.update({"current_step_status": "failed"})
            return ReplyResult(
                target=AgentTarget(admin_agent),
                message=f"Error reading reactor state: {e}. Need human intervention.",
                context_variables=context_variables
            )
            
           
    register_function(
        read_reactor_state,
        caller=diagnosis_agent.agent,
        executor=diagnosis_agent.agent,
        description="""
        Read the reactor state from a CSV file
        """,
    )
    
    register_function(
        update_reactor_state,
        caller=updater_agent.agent,
        executor=updater_agent.agent,
        description="""
        Update the reactor state with the desired action and store results as a CSV.
        Args:
            action (str): The action to be taken by the nuclear reactor
        """,
    )
    
    
    if namac_instance.is_planning_and_control:
        register_function(
            record_status,
            caller=control_agent.agent,
            executor=control_agent.agent,
            description=r"""
            Updates the context and returns the current progress.
            Must be called **before transitioning from the current task to the up-coming task**

            Args:
                current_step_status (str): The status of the current step ("complete", "failed").
                current_step_number (int): The number of the current step.
                next_task (str): Description of the next sub-task.
                current_instructions (str): Instructions for the sub-task.
                next_agent (str): The agent responsible for the next sub-task.
                context_variables (dict): context dictionary.

            Returns:
                ReplyResult: Contains a formatted status message and updated context.
            """,
        )
        
        register_function(
                extract_plan_from_planner,
                caller=planner_agent.agent,
                executor=planner_agent.agent,
                description="""
                Extract the plan from planner's messages and store it for control agent.
                Args:
                    plan (str): The plan to execute.
                """,
        )
        
