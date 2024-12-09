def mysplit(s):
    head = s.rstrip('0123456789')
    tail = s[len(head):]
    return head, tail