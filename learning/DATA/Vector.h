#pragma once
#include<iostream>
#include<cassert>
using namespace std;
template<typename Item>
class Vector
{
private:
	Item *m_data;
	int mi_capacity;
	int mi_count;
	void __resize__(int amount)
	{
		assert(amount >= mi_count);
		Item *NewData = new Item[amount];
		for (int i = 0; i < mi_capacity; i++)
			NewData[i] = m_data[i];
		delete[] m_data;
		m_data = NewData;
		mi_capacity = amount;
	}
public:
	Vector(int capacity = 10)
	{
		assert(capacity > 1);
		mi_capacity = capacity;
		m_data = new Item[capacity];
		mi_count = 0;
	}
	void PushBack(Item item)
	{
		if (mi_count >= mi_capacity)
			__resize__(2 * mi_capacity);
		m_data[mi_count++] = item;
	}
	Item Pop()
	{
		assert(mi_count > 0);
		T ret = m_data[--mi_count];
		if (size == mi_capacity / 4)
			__resize__(mi_capacity / 2);
		return ret;
	}
	~Vector() { delete[] m_data; }
	int GetCurrentCapacity() { return mi_capacity; }
	int Size() { return mi_count; }
};