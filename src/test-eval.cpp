#include <iostream>
#include "evaluation.h"

int main(int, char**) {

    auto cm = ConfusionMatrix{ 5 };
    cm.insert(0, 0);
    cm.insert(1, 1);
    cm.insert(2, 2);
    cm.insert(3, 3);
    cm.insert(4, 4);
    cm.insert(2, 1);

    std::cout << cm.accuracy() << std::endl;
    std::cout << cm.precision_recall_f1().average_fscore() << std::endl;
    return 0;
}
