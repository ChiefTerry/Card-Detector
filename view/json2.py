import json


class importJson:
    pass

    # def importJson(self, array_list):
    #     data = {}
    #     for i in range(len(array_list)):
    #         player_num = 'player' + str(i)
    #         player = array_list[i]
    #         data[player_num] = {}
    #         data[player_num]["cards"] = {}
    #         for f in range(len(array_list[i])):
    #             card = array_list[i][f]
    #             card_key = 'card' + str(i)
    #             data[player_num]["cards"][card_key] = {}
    #
    #             # add attribute of the cards
    #             data[player_num]["cards"][card_key]['x'] = card.get_x()
    #             data[player_num]["cards"][card_key]['y'] = card.get_y()
    #             data[player_num]["cards"][card_key]['type'] = card.get_type()
    #
    #         data[player_num]['heart'] = player.get_heart()
    #         data[player_num]['point'] = player.get_point()
    #         data[player_num]['bang_bullet'] = player.get_bang_bullet()
    #         data[player_num]['click_bullet'] = player.get_click_bullet()
    #         data[player_num]['num_diamond'] = player.get_num_diamond()
    #         data[player_num]['num_picture'] = player.get_num_picture()
    #         # data[player_num]['is_surrender'] = player.is_surrender()
    #         # data[player_num]['is_dead'] = player.is_dead()
    #
    #         # append card in list cards
    #
    #     with open('data.txt', 'w') as outfile:
    #         json.dump(data, outfile)
