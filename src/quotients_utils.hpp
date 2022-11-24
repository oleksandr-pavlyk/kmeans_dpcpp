// quotients utility

#pragma once

template <typename T>
T quotient_ceil(T n, T m) { return (n + m - 1) / m; }
