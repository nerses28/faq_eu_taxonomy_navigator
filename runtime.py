import sys

print('FAQs of the EU Taxonomy Navigator\n')
print('\033[1;33mChoose the model!\033[0m Options: [gpt4o_gen, gpt4o_choose, t5_gen, simple_faq0, simple_faq1, simple_faq2]')

model_choice = input('Enter your choice: ')

if model_choice == 'exit()':
    sys.exit()
elif model_choice in ["simple_faq0", "simple_faq1", "simple_faq2"]:
    if model_choice == "simple_faq0":
        from simple_faq import make_query0 as make_query
    elif model_choice == "simple_faq1":
        from simple_faq import make_query1 as make_query
    elif model_choice == "simple_faq2":
        from simple_faq import make_query2 as make_query

    while True:
        print('\n==================================================')
        print('\033[1;36mType "exit()" to stop\033[0m')
        query = input('\033[1;32mEnter you question:\033[0m ')
        if query == 'exit()':
            break
        else:
            answer, _ = make_query(query)
            print('\033[1;35mANSWER: \033[0m')
            print(answer)
elif model_choice in ["gpt4o_gen", "gpt4o_choose"]:
    api_key = input('Enter your OpenAI api_key: ')
    prompt_name = "prompt_choose"
    if model_choice == "gpt4o_gen":
        prompt_name = "prompt_gen"

    from top3 import make_query as get_top3
    from gpt4o_faq import make_query
    while True:
        print('\n==================================================')
        print('\033[1;36mType "exit()" to stop\033[0m')
        query = input('\033[1;32mEnter you question:\033[0m ')
        if query == 'exit()':
            break
        else:
            top3_ind = get_top3(query)
            answer = make_query(query, prompt_name, top3_ind, api_key)
            print('\033[1;35mANSWER: \033[0m')
            print(answer)
elif model_choice == 't5_gen':
    from top3 import make_query as get_top3
    from t5_faq import make_query

    while True:
        print('\n==================================================')
        print('\033[1;36mType "exit()" to stop\033[0m')
        query = input('\033[1;32mEnter you question:\033[0m ')
        if query == 'exit()':
            break
        else:
            top3_ind = get_top3(query)
            answer = make_query(query, list(reversed(top3_ind)))
            print('\033[1;35mANSWER: \033[0m')
            print(answer)
else:
    print('\033[1;31mERROR: INCORRECT MODEL\033[0m')
    sys.exit()


