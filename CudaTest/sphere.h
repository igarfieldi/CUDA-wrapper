#pragma once

#include "vec.h"

template < class T >
struct sphere {
	vec3<T> m_pos;
	T rad;
};