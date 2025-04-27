const { createClient } = require('@supabase/supabase-js');

// ✅ Replace with your actual Supabase project URL and anon key
const SUPABASE_URL = 'https://bnxzvzqapyiovxnimbtl.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJueHp2enFhcHlpb3Z4bmltYnRsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDUyNTY3NTYsImV4cCI6MjA2MDgzMjc1Nn0.2Stk0zVhwfXCPX1x8o5v2TOhqCytUFepFxbF5YO6Wgo';

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

async function getData() {
  const { data, error } = await supabase
    .from('resqflow')
    .select('*');

  if (error) {
    console.error('Error fetching data:', error);
  } else {
    console.log('Data:', data);
  }
}

getData();
