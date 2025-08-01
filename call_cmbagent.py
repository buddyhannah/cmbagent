import sys
from cmbagent import planning_and_control, one_shot

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <case_number>")
        sys.exit(1)

    try:
        case = int(sys.argv[1])
    except ValueError:
        print("case_number must be an integer.")
        sys.exit(1)

    if case == 1:
        task = open('prompts/prompt4.txt').read()
        results = planning_and_control(
            task=task,
            max_rounds_control=500,
            n_plan_reviews=1,
            max_n_attempts=4,
            max_plan_steps=7,
            plan_instructions=(
                "Use engineer agent for the whole analysis, and researcher at the very end "
                "in the last step to comment on results."
            )
        )
    elif case == 2:
        task = open('prompts/prompt.txt').read()
        results = one_shot(task=task, max_n_attempts = 5)
    else:
        print("Invalid case number")
        sys.exit(1)


if __name__ == "__main__":
    main()
