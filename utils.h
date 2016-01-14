#ifndef UTILS_H
#define UTILS_H
#include <opencv2/core/core.hpp>

namespace cmt{

float calcRectOverlap(cv::Rect r, cv::Rect rOther)
{
    int x0 = std::max(r.x , rOther.x);
    int x1 = std::min(r.x + r.width, rOther.x + rOther.width);
    int y0 = std::max(r.y, rOther.y);
    int y1 = std::min(r.y + r.height, rOther.y + rOther.height);

    if (x0 >= x1 || y0 >= y1) return 0.f;

    float areaInt = (x1-x0)*(y1-y0);
    return areaInt/((float)r.width*r.height+(float)rOther.width*rOther.height-areaInt);
}

template<class T>
inline T calcExpect(std::vector<T>& input)
{
    T e = 0;
    for(auto it = input.begin(); it!=input.end(); it++)
        e += *it;
    e /= input.size();
    return e;
}

template<class T>
T calcSigma(std::vector<T>& input)
{
    T e = calcExpect(input);
    T sigma = 0;
    for(auto it = input.begin(); it!=input.end(); it++)
        sigma += (*it-e)*(*it-e);
    sigma /= input.size();
    return sigma;
}

}
#endif // UTILS_H
